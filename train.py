"""
训练模块
========
包含所有训练相关函数：
  - 深度模型训练（MLP / CNN，带早停）
  - 自编码器训练（无监督）
  - 温度缩放 / 最优阈值搜索
  - 传统模型 + 集成（SVM / RF / LDA / Stacking）
  - 交叉验证
  - 样本量实验
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score)

from model import EnhancedMLP, MLPWrapper
from data_loader import select_features_with_rf


# ====================================================================
# 1. 深度学习训练
# ====================================================================

def train_dl_with_early_stopping(model, train_loader, val_loader, pos_weight=1.0,
                                 epochs=200, lr=0.001, weight_decay=5e-3,
                                 patience=15, min_delta=1e-4, device='cuda'):
    """训练 MLP / CNN 模型，带早停和学习率调度。

    参数
    ----------
    model : nn.Module
        PyTorch 模型实例。
    train_loader : DataLoader
    val_loader : DataLoader
    pos_weight : float
        正类权重（用于 BCELoss）。
    epochs : int
        最大训练轮数。
    lr : float
        初始学习率。
    weight_decay : float
        L2 正则化系数。
    patience : int
        早停耐心值。
    min_delta : float
        最小改善阈值。
    device : str
        计算设备。

    返回
    -------
    model, train_losses, val_losses
    """
    model.to(device)
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.3)

    best_val_loss = float('inf')
    wait = 0
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device).float().view(-1, 1)
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


# ====================================================================
# 2. 自编码器训练（无监督）
# ====================================================================

def train_autoencoder(model, train_loader, val_loader=None, epochs=100, lr=0.001,
                      weight_decay=1e-4, device='cuda'):
    """训练自编码器，使用 MSE 重构损失。

    参数
    ----------
    model : Autoencoder
    train_loader : DataLoader
        训练数据加载器（忽略标签）。
    val_loader : DataLoader 或 None
    epochs : int
    lr : float
    weight_decay : float
    device : str

    返回
    -------
    model : Autoencoder
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    best_state = None
    patience = 10
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for Xb, _ in train_loader:  # 忽略标签
            Xb = Xb.to(device)
            optimizer.zero_grad()
            decoded, _ = model(Xb)
            loss = criterion(decoded, Xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        # 验证损失
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, _ in val_loader:
                    Xb = Xb.to(device)
                    decoded, _ = model(Xb)
                    loss = criterion(decoded, Xb)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"AE Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_state = model.state_dict().copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        else:
            print(f"AE Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = model.state_dict().copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ====================================================================
# 3. 温度缩放与阈值选择
# ====================================================================

def learn_temperature(logits, labels, init_T=1.0, lr=0.01, epochs=80):
    """学习最优温度缩放参数 T。

    参数
    ----------
    logits : np.ndarray
        模型原始 logits。
    labels : np.ndarray
        真实标签。
    init_T : float
        初始温度。
    lr : float
        学习率。
    epochs : int

    返回
    -------
    best_T : float
    """
    logits_t = torch.tensor(logits, dtype=torch.float32).view(-1, 1)
    labels_t = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    T = nn.Parameter(torch.tensor(init_T, dtype=torch.float32))
    optimizer = optim.Adam([T], lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    best_T = init_T
    for epoch in range(epochs):
        optimizer.zero_grad()
        T_clamped = torch.clamp(T, 0.1, 10.0)
        scaled_logits = logits_t / T_clamped
        loss = criterion(scaled_logits, labels_t)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            T.clamp_(0.1, 10.0)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_T = T.item()
        if (epoch + 1) % 20 == 0:
            print(f"Temp Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, T: {T.item():.4f}")
    print(f"最佳温度 T = {best_T:.4f} (损失 {best_loss:.4f})")
    return best_T


def find_optimal_thresholds(y_true, y_proba):
    """基于 PR 曲线寻找最大化 F1 的阈值。

    参数
    ----------
    y_true : np.ndarray
    y_proba : np.ndarray

    返回
    -------
    best_f1_thr : float
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thr = thresholds_pr[best_f1_idx] if best_f1_idx < len(thresholds_pr) else 0.5
    print(f"F1 最大化阈值: {best_f1_thr:.4f} (F1: {f1_scores[best_f1_idx]:.4f})")
    return best_f1_thr


# ====================================================================
# 4. 传统模型 + 集成
# ====================================================================

def train_traditional_and_ensemble(X_train, y_train, X_val, y_val, X_test, y_test,
                                   mlp_model, mlp_temp, mlp_thr,
                                   cnn_model, cnn_temp, cnn_thr,
                                   ae_model):
    """训练传统模型（SVM / RF / LDA）并与深度学习模型集成。

    返回
    -------
    results : dict
        各基础模型结果。
    weighted_proba : np.ndarray
    weighted_pred : np.ndarray
    weighted_scores : tuple (acc, auc, f1)
    stacking : StackingClassifier
    stack_proba : np.ndarray
    stack_pred : np.ndarray
    stack_scores : tuple (acc, auc, f1)
    time_info : dict
    """
    device = mlp_model.net[0].weight.device if hasattr(mlp_model, 'net') else 'cuda'

    models = {
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'RF': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, class_weight='balanced'),
        'LDA': LinearDiscriminantAnalysis()
    }
    results = {}
    val_aucs = []
    proba_list = []
    model_names = []
    time_info = {}

    # ----- 传统模型 -----
    for name, clf in models.items():
        start_fit = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - start_fit

        start_pred = time.time()
        y_proba_val = clf.predict_proba(X_val)[:, 1]
        y_proba_test = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        pred_time = time.time() - start_pred

        time_info[name] = {'fit_time': fit_time, 'predict_time': pred_time}

        val_auc = roc_auc_score(y_val, y_proba_val)
        val_aucs.append(val_auc)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba_test)
        results[name] = {'model': clf, 'accuracy': acc, 'auc': auc,
                         'y_pred': y_pred, 'y_proba': y_proba_test}
        proba_list.append(y_proba_test)
        model_names.append(name)
        print(f"{name:15s} Test Acc: {acc:.4f}, Test AUC: {auc:.4f}")

    # ----- MLP -----
    mlp_wrapper = MLPWrapper(mlp_model, temperature=mlp_temp, threshold=mlp_thr, device=device)
    mlp_wrapper.fit(X_train, y_train)
    start_pred = time.time()
    y_proba_val_mlp = mlp_wrapper.predict_proba(X_val)[:, 1]
    y_proba_mlp = mlp_wrapper.predict_proba(X_test)[:, 1]
    y_pred_mlp = mlp_wrapper.predict(X_test)
    pred_time = time.time() - start_pred
    time_info['MLP'] = {'fit_time': None, 'predict_time': pred_time}

    val_auc_mlp = roc_auc_score(y_val, y_proba_val_mlp)
    val_aucs.append(val_auc_mlp)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    auc_mlp = roc_auc_score(y_test, y_proba_mlp)
    results['MLP'] = {'model': mlp_wrapper, 'accuracy': acc_mlp, 'auc': auc_mlp,
                      'y_pred': y_pred_mlp, 'y_proba': y_proba_mlp}
    proba_list.append(y_proba_mlp)
    model_names.append('MLP')
    print(f"MLP{' ' * 12} Test Acc: {acc_mlp:.4f}, Test AUC: {auc_mlp:.4f}")

    # ----- CNN -----
    cnn_wrapper = MLPWrapper(cnn_model, temperature=cnn_temp, threshold=cnn_thr, device=device)
    cnn_wrapper.fit(X_train, y_train)
    start_pred = time.time()
    y_proba_val_cnn = cnn_wrapper.predict_proba(X_val)[:, 1]
    y_proba_cnn = cnn_wrapper.predict_proba(X_test)[:, 1]
    y_pred_cnn = cnn_wrapper.predict(X_test)
    pred_time = time.time() - start_pred
    time_info['CNN'] = {'fit_time': None, 'predict_time': pred_time}

    val_auc_cnn = roc_auc_score(y_val, y_proba_val_cnn)
    val_aucs.append(val_auc_cnn)
    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    auc_cnn = roc_auc_score(y_test, y_proba_cnn)
    results['CNN'] = {'model': cnn_wrapper, 'accuracy': acc_cnn, 'auc': auc_cnn,
                      'y_pred': y_pred_cnn, 'y_proba': y_proba_cnn}
    proba_list.append(y_proba_cnn)
    model_names.append('CNN')
    print(f"CNN{' ' * 12} Test Acc: {acc_cnn:.4f}, Test AUC: {auc_cnn:.4f}")

    # ----- AE+LR -----
    ae_model.fit(X_train, y_train)
    start_pred = time.time()
    y_proba_val_ae = ae_model.predict_proba(X_val)[:, 1]
    y_proba_ae = ae_model.predict_proba(X_test)[:, 1]
    y_pred_ae = ae_model.predict(X_test)
    pred_time = time.time() - start_pred
    time_info['AE+LR'] = {'fit_time': None, 'predict_time': pred_time}

    val_auc_ae = roc_auc_score(y_val, y_proba_val_ae)
    val_aucs.append(val_auc_ae)
    acc_ae = accuracy_score(y_test, y_pred_ae)
    auc_ae = roc_auc_score(y_test, y_proba_ae)
    results['AE+LR'] = {'model': ae_model, 'accuracy': acc_ae, 'auc': auc_ae,
                        'y_pred': y_pred_ae, 'y_proba': y_proba_ae}
    proba_list.append(y_proba_ae)
    model_names.append('AE+LR')
    print(f"AE+LR{' ' * 9} Test Acc: {acc_ae:.4f}, Test AUC: {auc_ae:.4f}")

    # ----- 加权平均（无训练）-----
    total = sum(val_aucs)
    weights = [a / total for a in val_aucs]
    weighted_proba = np.zeros_like(proba_list[0])
    for proba, w in zip(proba_list, weights):
        weighted_proba += w * proba
    weighted_pred = (weighted_proba >= 0.5).astype(int)
    weighted_acc = accuracy_score(y_test, weighted_pred)
    weighted_auc = roc_auc_score(y_test, weighted_proba)
    weighted_f1 = f1_score(y_test, weighted_pred)
    print(f"Weighted Avg (SVM+RF+LDA+MLP+CNN+AE): Acc={weighted_acc:.4f}, AUC={weighted_auc:.4f}, F1={weighted_f1:.4f}")
    print(f"  权重: SVM={weights[0]:.3f}, RF={weights[1]:.3f}, LDA={weights[2]:.3f}, MLP={weights[3]:.3f}, CNN={weights[4]:.3f}, AE={weights[5]:.3f}")

    # ----- Stacking -----
    base_learners = [
        ('svm', SVC(probability=True, random_state=42, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, class_weight='balanced')),
        ('lda', LinearDiscriminantAnalysis()),
        ('mlp', MLPWrapper(mlp_model, temperature=mlp_temp, threshold=mlp_thr, device=device)),
        ('cnn', MLPWrapper(cnn_model, temperature=cnn_temp, threshold=cnn_thr, device=device)),
        ('ae', ae_model),
    ]
    stacking = StackingClassifier(estimators=base_learners,
                                  final_estimator=LogisticRegression(class_weight='balanced', C=0.1),
                                  cv=5, passthrough=True)

    start_fit = time.time()
    stacking.fit(X_train, y_train)
    fit_time = time.time() - start_fit
    start_pred = time.time()
    y_pred_stack = stacking.predict(X_test)
    y_proba_stack = stacking.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start_pred
    time_info['Stacking'] = {'fit_time': fit_time, 'predict_time': pred_time}

    stack_acc = accuracy_score(y_test, y_pred_stack)
    stack_auc = roc_auc_score(y_test, y_proba_stack)
    stack_f1 = f1_score(y_test, y_pred_stack)
    print(f"Stacking (SVM+RF+LDA+MLP+CNN+AE, passthrough): Acc={stack_acc:.4f}, AUC={stack_auc:.4f}, F1={stack_f1:.4f}")

    return results, weighted_proba, weighted_pred, (weighted_acc, weighted_auc, weighted_f1), \
           stacking, y_proba_stack, y_pred_stack, (stack_acc, stack_auc, stack_f1), time_info


# ====================================================================
# 5. 样本量实验
# ====================================================================

def sample_size_experiment(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
                           proportions=None, n_features=180, device='cuda',
                           pos_weight=1.0, weight_decay=5e-3, dropout_rate=0.5):
    """不同训练样本比例下的 MLP 性能实验。

    返回 pd.DataFrame，包含各比例下的验证/测试指标和训练时间。
    """
    import pandas as pd
    from evaluate import evaluate_dl_logits, evaluate_dl

    if proportions is None:
        proportions = np.linspace(0.1, 1.0, 10)

    results = {'proportion': [], 'train_size': [],
               'val_acc': [], 'val_auc': [],
               'test_acc': [], 'test_auc': [],
               'train_time': []}

    for prop in proportions:
        if prop == 1.0:
            X_sub = X_train_raw
            y_sub = y_train
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=prop, random_state=42)
            for train_idx, _ in sss.split(X_train_raw, y_train):
                X_sub = X_train_raw[train_idx]
                y_sub = y_train[train_idx]

        print(f"\n--- 样本比例 {prop:.0%}, 训练样本数 {len(y_sub)}, 固定特征数 {n_features} ---")

        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_val_scaled = scaler.transform(X_val_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        selector_f = SelectKBest(f_classif, k=1000)
        X_sub_f = selector_f.fit_transform(X_sub_scaled, y_sub)
        X_val_f = selector_f.transform(X_val_scaled)
        X_test_f = selector_f.transform(X_test_scaled)

        X_sub_rf, X_val_rf, X_test_rf, _, _ = select_features_with_rf(
            X_sub_f, y_sub, X_val_f, X_test_f, n_features=n_features
        )

        train_ds = TensorDataset(torch.tensor(X_sub_rf, dtype=torch.float32),
                                 torch.tensor(y_sub, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val_rf, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test_rf, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        model = EnhancedMLP(input_dim=n_features, hidden_dims=[64, 32], dropout_rate=dropout_rate)
        start_time = time.time()
        model, _, _ = train_dl_with_early_stopping(
            model, train_loader, val_loader, pos_weight=pos_weight,
            epochs=100, patience=15, device=device, weight_decay=weight_decay, min_delta=1e-4
        )
        train_time_elapsed = time.time() - start_time

        logits_val, y_val_np = evaluate_dl_logits(model, val_loader, device)
        T = learn_temperature(logits_val, y_val_np, init_T=1.0, lr=0.01, epochs=80)
        _, y_proba_val_raw, _ = evaluate_dl(model, val_loader, device, threshold=0.5, temperature=T)
        best_thr = find_optimal_thresholds(y_val, y_proba_val_raw)
        _, y_proba_val, _ = evaluate_dl(model, val_loader, device, threshold=best_thr, temperature=T)
        _, y_proba_test, _ = evaluate_dl(model, test_loader, device, threshold=best_thr, temperature=T)

        val_acc = accuracy_score(y_val, (y_proba_val >= 0.5).astype(int))
        val_auc = roc_auc_score(y_val, y_proba_val)
        test_acc = accuracy_score(y_test, (y_proba_test >= 0.5).astype(int))
        test_auc = roc_auc_score(y_test, y_proba_test)

        results['proportion'].append(prop)
        results['train_size'].append(len(y_sub))
        results['val_acc'].append(val_acc)
        results['val_auc'].append(val_auc)
        results['test_acc'].append(test_acc)
        results['test_auc'].append(test_auc)
        results['train_time'].append(train_time_elapsed)

        print(f"Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}, Test Acc={test_acc:.4f}, Test AUC={test_auc:.4f}, Time={train_time_elapsed:.1f}s")

    return pd.DataFrame(results)


# ====================================================================
# 6. 交叉验证
# ====================================================================

def cross_val_dl_enhanced(X_raw, y, n_splits=5, pos_weight=1.0, weight_decay=5e-3,
                          n_features_f=1000, n_features_rf=180, dropout_rate=0.5,
                          device='cuda'):
    """使用 StratifiedKFold 对 MLP 进行交叉验证。

    返回各折的 acc / auc / f1 列表。
    """
    from evaluate import evaluate_dl_logits, evaluate_dl

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, auc_list, f1_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        print(f"\nFold {fold+1}/{n_splits}")
        X_tr_raw, X_va_raw = X_raw[train_idx], X_raw[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_va_scaled = scaler.transform(X_va_raw)

        selector_f = SelectKBest(f_classif, k=n_features_f)
        X_tr_f = selector_f.fit_transform(X_tr_scaled, y_tr)
        X_va_f = selector_f.transform(X_va_scaled)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr_f, y_tr)
        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:n_features_rf]
        X_tr_rf = X_tr_f[:, sorted_idx]
        X_va_rf = X_va_f[:, sorted_idx]

        print(f"  训练集: {X_tr_rf.shape}, 验证集: {X_va_rf.shape}")

        train_ds = TensorDataset(torch.tensor(X_tr_rf, dtype=torch.float32),
                                 torch.tensor(y_tr, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_va_rf, dtype=torch.float32),
                               torch.tensor(y_va, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = EnhancedMLP(input_dim=X_tr_rf.shape[1], hidden_dims=[64, 32], dropout_rate=dropout_rate)
        model, _, _ = train_dl_with_early_stopping(
            model, train_loader, val_loader, pos_weight=pos_weight,
            epochs=150, patience=15, device=device, weight_decay=weight_decay, min_delta=1e-4
        )

        logits_val, y_val_np = evaluate_dl_logits(model, val_loader, device)
        T = learn_temperature(logits_val, y_val_np, init_T=1.0, lr=0.01, epochs=80)
        _, y_proba_val, _ = evaluate_dl(model, val_loader, device, threshold=0.5, temperature=T)
        best_thr = find_optimal_thresholds(y_va, y_proba_val)
        y_pred, y_proba, _ = evaluate_dl(model, val_loader, device, threshold=best_thr, temperature=T)
        acc = accuracy_score(y_va, y_pred)
        auc = roc_auc_score(y_va, y_proba)
        f1 = f1_score(y_va, y_pred)
        acc_list.append(acc); auc_list.append(auc); f1_list.append(f1)
        print(f"Fold {fold+1} Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, T={T:.3f}, thr={best_thr:.3f}")

    print(f"\nCV 平均: Acc={np.mean(acc_list):.4f}±{np.std(acc_list):.4f}, AUC={np.mean(auc_list):.4f}±{np.std(auc_list):.4f}, F1={np.mean(f1_list):.4f}±{np.std(f1_list):.4f}")
    return acc_list, auc_list, f1_list
