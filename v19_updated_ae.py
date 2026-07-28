
"""
深度学习课程项目 
基于深度学习的基因表达谱数据疾病分类
数据集：GSE25066 (乳腺癌新辅助化疗 pCR vs RD)


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, 
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time
import joblib
import os
import re

# 尝试导入 pycombat（可选）
try:
    from pycombat import pycombat
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False
    print("pycombat 未安装，批次校正将被跳过。如需使用请运行: pip install pycombat")

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ====================================================================
# 1. 数据加载与注释解析
# ====================================================================
def parse_gse25066_series_matrix(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    sample_ids = None
    labels = {}
    batch_labels = {}
    expr_lines = []
    header = None
    in_data = False
    label_pattern = re.compile(r'pathologic_response_pcr_rd', re.IGNORECASE)
    batch_pattern = re.compile(r'batch', re.IGNORECASE)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('!'):
            if line.startswith('!Sample_geo_accession'):
                parts = line.split('\t')
                sample_ids = [x for x in parts[1:] if x != '']
                print(f"发现 {len(sample_ids)} 个样本")
            elif label_pattern.search(line):
                parts = line.split('\t')
                for i, val in enumerate(parts[1:], start=0):
                    if i >= len(sample_ids):
                        break
                    val_clean = val.strip('"').strip()
                    if 'pCR' in val_clean:
                        labels[sample_ids[i]] = 'pCR'
                    elif 'RD' in val_clean:
                        labels[sample_ids[i]] = 'RD'
            elif batch_pattern.search(line) and '!Sample_characteristics_ch1' in line:
                parts = line.split('\t')
                for i, val in enumerate(parts[1:], start=0):
                    if i >= len(sample_ids):
                        break
                    val_clean = val.strip('"').strip()
                    if ':' in val_clean:
                        batch_val = val_clean.split(':')[-1].strip()
                        batch_labels[sample_ids[i]] = batch_val
            if '!series_matrix_table_begin' in line:
                in_data = True
                continue
        else:
            if in_data:
                if header is None:
                    header = line.split('\t')
                else:
                    expr_lines.append(line.split('\t'))
    if header is None or len(expr_lines) == 0:
        raise ValueError("未能解析表达矩阵，请检查文件格式")
    df_expr = pd.DataFrame(expr_lines, columns=header)
    df_expr.set_index(header[0], inplace=True)
    df_expr = df_expr.apply(pd.to_numeric, errors='coerce')
    common_samples = [s for s in sample_ids if s in labels and s in df_expr.columns]
    df_expr = df_expr[common_samples]
    labels = {k: v for k, v in labels.items() if k in common_samples}
    if batch_labels:
        batch_labels = {k: v for k, v in batch_labels.items() if k in common_samples}
    y_series = pd.Series(labels).reindex(df_expr.columns)
    le = LabelEncoder()
    y = le.fit_transform(y_series)
    print(f"成功提取 {len(y)} 个样本标签：{dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"pCR: {sum(y==0)}, RD: {sum(y==1)}")
    if batch_labels:
        print(f"检测到批次信息，共 {len(set(batch_labels.values()))} 个批次")
    else:
        print("未检测到批次信息")
    return df_expr, y, le.classes_, batch_labels if batch_labels else None

def parse_gpl96_annotation(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    header = None
    data = []
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        if header is None:
            header = line.split('\t')
            for idx, col in enumerate(header):
                if col.lower() == 'gene symbol':
                    header[idx] = 'Gene Symbol'
                    break
        else:
            data.append(line.split('\t'))
    df_anno = pd.DataFrame(data, columns=header)
    if 'Gene Symbol' not in df_anno.columns:
        for col in df_anno.columns:
            if 'symbol' in col.lower():
                df_anno.rename(columns={col: 'Gene Symbol'}, inplace=True)
                break
    if 'ID' not in df_anno.columns:
        for col in df_anno.columns:
            if col.lower() == 'id':
                df_anno.rename(columns={col: 'ID'}, inplace=True)
                break
    probe2symbol = df_anno.set_index('ID')['Gene Symbol'].to_dict()
    probe2symbol = {k: v for k, v in probe2symbol.items() if pd.notna(v) and v != ''}
    print(f"加载注释：共 {len(probe2symbol)} 个探针-基因映射")
    return probe2symbol

# ====================================================================
# 2. 基础预处理
# ====================================================================
def basic_preprocess(expr_df, variance_thresh=0.5):
    missing_ratio = expr_df.isnull().mean(axis=1)
    keep = missing_ratio <= 0.2
    expr_df = expr_df[keep]
    print(f"缺失率>20%的探针移除后，剩余 {expr_df.shape[0]} 个探针")
    imputer = SimpleImputer(strategy='median')
    expr_imputed = pd.DataFrame(imputer.fit_transform(expr_df),
                                index=expr_df.index, columns=expr_df.columns)
    scaler = StandardScaler()
    expr_scaled = pd.DataFrame(scaler.fit_transform(expr_imputed.T).T,
                               index=expr_imputed.index, columns=expr_imputed.columns)
    var_selector = VarianceThreshold(threshold=variance_thresh)
    var_selector.fit(expr_scaled.T)
    keep_idx = var_selector.get_support()
    expr_var_df = expr_scaled[keep_idx]
    print(f"方差过滤后保留 {expr_var_df.shape[0]} 个探针")
    return expr_var_df, var_selector, scaler, imputer

# ====================================================================
# 3. 批次效应校正
# ====================================================================
def combat_correction(expr_df, batch_series):
    if not COMBAT_AVAILABLE:
        print("pycombat 未安装，跳过批次校正。")
        return expr_df
    samples = expr_df.columns
    batch_list = [batch_series[s] for s in samples]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    batch_encoded = le.fit_transform(batch_list)
    data_corrected = pycombat(expr_df.values, batch_encoded)
    return pd.DataFrame(data_corrected, index=expr_df.index, columns=expr_df.columns)

# ====================================================================
# 4. RF 二次特征选择
# ====================================================================
def select_features_with_rf(X_train, y_train, X_val, X_test=None, n_features=180):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:n_features]
    X_train_sel = X_train[:, sorted_idx]
    X_val_sel = X_val[:, sorted_idx]
    X_test_sel = X_test[:, sorted_idx] if X_test is not None else None
    return X_train_sel, X_val_sel, X_test_sel, sorted_idx, rf

# ====================================================================
# 5. 深度学习模型（MLP, CNN, 自编码器）
# ====================================================================
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.5):
        super(EnhancedMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EnhancedCNN(nn.Module):
    def __init__(self, input_dim, n_conv_layers=2, n_filters=32, kernel_size=5,
                 hidden_dims=[128, 64], dropout_rate=0.5):
        super(EnhancedCNN, self).__init__()
        conv_layers = []
        in_channels = 1
        for i in range(n_conv_layers):
            out_channels = n_filters * (2 ** i)
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.conv_block = nn.Sequential(*conv_layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.conv_block(dummy)
            conv_out_dim = out.view(1, -1).shape[1]
        fc_layers = []
        prev_dim = conv_out_dim
        for h_dim in hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, h_dim))
            fc_layers.append(nn.BatchNorm1d(h_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        fc_layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ========== 自编码器 ==========
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64, hidden_dims=[128, 64]):
        """
        自编码器：编码器将input_dim压缩到encoding_dim，解码器恢复。
        """
        super(Autoencoder, self).__init__()
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器（对称结构）
        decoder_layers = []
        prev_dim = encoding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded  # 返回重建和编码特征

    def encode(self, x):
        return self.encoder(x)

# ====================================================================
# 6. 训练与评估函数（适用于MLP, CNN, 自编码器）
# ====================================================================
def train_dl_with_early_stopping(model, train_loader, val_loader, pos_weight=1.0, 
                                 epochs=200, lr=0.001, weight_decay=5e-3, 
                                 patience=15, min_delta=1e-4, device='cuda'):
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

        if (epoch+1) % 30 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def evaluate_dl_logits(model, loader, device='cuda'):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb).cpu().numpy().flatten()
            logits_list.extend(logits)
            labels_list.extend(yb.numpy())
    return np.array(logits_list), np.array(labels_list)

def evaluate_dl(model, loader, device='cuda', threshold=0.5, temperature=1.0):
    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb).cpu().numpy().flatten()
            calibrated_logits = logits / temperature
            prob = 1 / (1 + np.exp(-calibrated_logits))
            pred = (prob >= threshold).astype(int)
            preds.extend(pred)
            probs.extend(prob)
            labels.extend(yb.numpy())
    return np.array(preds), np.array(probs), np.array(labels)

# ========== 自编码器训练函数（无监督） ==========
def train_autoencoder(model, train_loader, val_loader=None, epochs=100, lr=0.001, 
                      weight_decay=1e-4, device='cuda'):
    """训练自编码器，使用MSE重构损失"""
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
# 7. 温度缩放与阈值选择
# ====================================================================
def learn_temperature(logits, labels, init_T=1.0, lr=0.01, epochs=80):
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
        if (epoch+1) % 20 == 0:
            print(f"Temp Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, T: {T.item():.4f}")
    print(f"最佳温度 T = {best_T:.4f} (损失 {best_loss:.4f})")
    return best_T

def find_optimal_thresholds(y_true, y_proba):
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_thr = thresholds_pr[best_f1_idx] if best_f1_idx < len(thresholds_pr) else 0.5
    print(f"F1最大化阈值: {best_f1_thr:.4f} (F1: {f1_scores[best_f1_idx]:.4f})")
    return best_f1_thr

# ====================================================================
# 8. MLPWrapper（通用，适用于MLP和CNN，也适用于自编码器+分类器）
# ====================================================================
class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, temperature=1.0, threshold=0.5, device='cuda'):
        self.model = model
        self.temperature = temperature
        self.threshold = threshold
        self.device = device
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("MLPWrapper仅支持二分类")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t).cpu().numpy().flatten()
            calibrated = logits / self.temperature
            prob_pos = 1 / (1 + np.exp(-calibrated))
            prob_neg = 1 - prob_pos
            return np.stack([prob_neg, prob_pos], axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

# ========== 自编码器分类器（AE + LogisticRegression） ==========
class AutoencoderClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, encoding_dim=64, hidden_dims=[128, 64], 
                 ae_epochs=100, ae_lr=0.001, ae_weight_decay=1e-4,
                 classifier=None, device='cuda'):
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_weight_decay = ae_weight_decay
        self.device = device
        self.autoencoder = None
        self.classifier = classifier if classifier else LogisticRegression(class_weight='balanced')
        self.input_dim = None
        self.classes_ = None

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        ae = Autoencoder(input_dim=self.input_dim, 
                         encoding_dim=self.encoding_dim, 
                         hidden_dims=self.hidden_dims)
        train_ds = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                 torch.zeros(len(X), dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        ae = train_autoencoder(ae, train_loader, epochs=self.ae_epochs, 
                               lr=self.ae_lr, weight_decay=self.ae_weight_decay, 
                               device=self.device)
        self.autoencoder = ae

        with torch.no_grad():
            X_encoded = ae.encode(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        self.classifier.fit(X_encoded, y)
        self.classes_ = np.unique(y)  # 关键修复：设置类别属性
        return self

    def predict_proba(self, X):
        if self.autoencoder is None:
            raise RuntimeError("模型未训练，请先调用 fit。")
        with torch.no_grad():
            X_encoded = self.autoencoder.encode(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return self.classifier.predict_proba(X_encoded)

    def predict(self, X):
        if self.autoencoder is None:
            raise RuntimeError("模型未训练，请先调用 fit。")
        with torch.no_grad():
            X_encoded = self.autoencoder.encode(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return self.classifier.predict(X_encoded)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# ====================================================================
# 9. 传统模型与集成
# ====================================================================
def train_traditional_and_ensemble(X_train, y_train, X_val, y_val, X_test, y_test,
                                   mlp_model, mlp_temp, mlp_thr,
                                   cnn_model, cnn_temp, cnn_thr,
                                   ae_model):
    """
    训练传统模型、MLP、CNN、AE+LR，并集成。
    返回结果字典、集成预测、以及时间字典。
    """
    models = {
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'RF': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, class_weight='balanced'),
        'LDA': LinearDiscriminantAnalysis()
    }
    results = {}
    val_aucs = []
    proba_list = []
    model_names = []
    time_info = {}  # 存储各模型时间

    # ----- 传统模型 -----
    for name, clf in models.items():
        # 训练计时
        start_fit = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - start_fit

        # 预测计时（包括验证集和测试集，但这里统一定义为测试集预测时间）
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
    mlp_wrapper.fit(X_train, y_train)  # 仅设置类别
    start_pred = time.time()
    y_proba_val_mlp = mlp_wrapper.predict_proba(X_val)[:, 1]
    y_proba_mlp = mlp_wrapper.predict_proba(X_test)[:, 1]
    y_pred_mlp = mlp_wrapper.predict(X_test)
    pred_time = time.time() - start_pred
    time_info['MLP'] = {'fit_time': None, 'predict_time': pred_time}  # fit_time 已在外部记录

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
    ae_model.fit(X_train, y_train)  # 确保训练
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
    weights = [a/total for a in val_aucs]
    weighted_proba = np.zeros_like(proba_list[0])
    for proba, w in zip(proba_list, weights):
        weighted_proba += w * proba
    weighted_pred = (weighted_proba >= 0.5).astype(int)
    weighted_acc = accuracy_score(y_test, weighted_pred)
    weighted_auc = roc_auc_score(y_test, weighted_proba)
    weighted_f1 = f1_score(y_test, weighted_pred)
    print(f"Weighted Avg (SVM+RF+LDA+MLP+CNN+AE): Acc={weighted_acc:.4f}, AUC={weighted_auc:.4f}, F1={weighted_f1:.4f}")
    print(f"  权重: SVM={weights[0]:.3f}, RF={weights[1]:.3f}, LDA={weights[2]:.3f}, MLP={weights[3]:.3f}, CNN={weights[4]:.3f}, AE={weights[5]:.3f}")

    # ----- Stacking（训练+预测计时）-----
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
# 10. 样本量实验（可保持不变，只针对MLP）
# ====================================================================
def sample_size_experiment(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
                           proportions=None, n_features=180, device='cpua',
                           pos_weight=1.0, weight_decay=5e-3, dropout_rate=0.5):
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

        model = EnhancedMLP(input_dim=n_features, hidden_dims=[64,32], dropout_rate=dropout_rate)
        start_time = time.time()
        model, _, _ = train_dl_with_early_stopping(
            model, train_loader, val_loader, pos_weight=pos_weight,
            epochs=100, patience=15, device=device, weight_decay=weight_decay, min_delta=1e-4
        )
        train_time = time.time() - start_time

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
        results['train_time'].append(train_time)

        print(f"Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}, Test Acc={test_acc:.4f}, Test AUC={test_auc:.4f}, Time={train_time:.1f}s")

    return pd.DataFrame(results)

# ====================================================================
# 11. 绘图函数
# ====================================================================
def plot_sample_size_curve(df_results, save_path='sample_size_vs_performance_v19_updated_ae.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：Accuracy
    ax = axes[0]
    ax.plot(df_results['train_size'], df_results['val_acc'], 'o-', label='Validation Acc')
    ax.plot(df_results['train_size'], df_results['test_acc'], 's-', label='Test Acc')
    for x, y in zip(df_results['train_size'], df_results['val_acc']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 5),
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    for x, y in zip(df_results['train_size'], df_results['test_acc']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 5),
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Sample Size vs Accuracy')
    ax.legend()
    ax.grid(True)

    # 右图：AUC
    ax = axes[1]
    ax.plot(df_results['train_size'], df_results['val_auc'], 'o-', label='Validation AUC')
    ax.plot(df_results['train_size'], df_results['test_auc'], 's-', label='Test AUC')
    for x, y in zip(df_results['train_size'], df_results['val_auc']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 5),
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    for x, y in zip(df_results['train_size'], df_results['test_auc']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 5),
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('AUC')
    ax.set_title('Sample Size vs AUC')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"样本量曲线已保存至 {save_path}")
# ==========  Part 1：主要性能对比 ==========
def plot_final_results_part1(trad_results, mlp_metrics, cnn_metrics, ae_metrics, weighted_metrics, stack_metrics,
                             train_losses, val_losses, y_train, y_val, y_test,
                             y_proba_mlp, y_proba_cnn, y_proba_ae, weighted_proba, stack_proba,
                             important_info, cv_metrics, best_threshold_mlp, best_threshold_cnn):
    from sklearn.metrics import recall_score

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # ---------- 1. ROC曲线（仅使用 trad_results + 集成模型） ----------
    # 从 trad_results 中绘制所有基础模型
    for name, res in trad_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", lw=1.5)
    # 添加集成模型（WeightedAvg 和 Stacking）
    fpr, tpr, _ = roc_curve(y_test, weighted_proba)
    ax1.plot(fpr, tpr, label=f"Weighted Avg (AUC={weighted_metrics['auc']:.3f})", lw=2, color='gold')
    fpr, tpr, _ = roc_curve(y_test, stack_proba)
    ax1.plot(fpr, tpr, label=f"Stacking (AUC={stack_metrics['auc']:.3f})", lw=2, color='magenta')
    ax1.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - All Models')
    ax1.legend(loc='lower right', fontsize=8)

    # ---------- 2. 学习曲线 ----------
    ax2.plot(train_losses, label='Training Loss')
    ax2.plot(val_losses, label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Learning Curves (MLP, thr={best_threshold_mlp:.3f})')
    ax2.legend()

    # ---------- 3. 性能对比条形图（Accuracy + AUC） ----------
    # 基础模型来自 trad_results，再加上集成模型
    base_names = list(trad_results.keys())          # ['SVM','RF','LDA','MLP','CNN','AE+LR']
    base_accs = [res['accuracy'] for res in trad_results.values()]
    base_aucs = [res['auc'] for res in trad_results.values()]
    # 集成模型
    ensemble_names = ['WeightedAvg', 'Stacking']
    ensemble_accs = [weighted_metrics['accuracy'], stack_metrics['accuracy']]
    ensemble_aucs = [weighted_metrics['auc'], stack_metrics['auc']]

    names = base_names + ensemble_names
    accs = base_accs + ensemble_accs
    aucs = base_aucs + ensemble_aucs

    x = np.arange(len(names))
    width = 0.35
    bars_acc = ax3.bar(x - width/2, accs, width, label='Accuracy', color='skyblue')
    bars_auc = ax3.bar(x + width/2, aucs, width, label='AUC', color='lightcoral')
    for bar, val in zip(bars_acc, accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars_auc, aucs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15, fontsize=8)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance (Accuracy & AUC)')
    ax3.legend()

    # ---------- 4. F1 分数 + 召回率 ----------
    # 同样使用统一数据源
    base_preds = [res['y_pred'] for res in trad_results.values()]
    ensemble_preds = [weighted_metrics['y_pred'], stack_metrics['y_pred']]
    all_preds = base_preds + ensemble_preds
    all_names = base_names + ensemble_names

    f1_scores = [f1_score(y_test, pred) for pred in all_preds]
    recall_scores = [recall_score(y_test, pred) for pred in all_preds]
    x2 = np.arange(len(all_names))
    width2 = 0.35
    bars_f1 = ax4.bar(x2 - width2/2, f1_scores, width2, label='F1 Score', color='mediumseagreen')
    bars_rec = ax4.bar(x2 + width2/2, recall_scores, width2, label='Recall', color='orange')
    for bar, val in zip(bars_f1, f1_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars_rec, recall_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax4.set_xticks(x2)
    ax4.set_xticklabels(all_names, rotation=15, fontsize=8)
    ax4.set_ylim(0, 1.1)
    ax4.set_ylabel('Score')
    ax4.set_title('F1 Score and Recall')
    ax4.legend()

    # ---------- 5. 重要基因 ----------
    symbols, importances = important_info
    ax5.barh(symbols, importances)
    ax5.set_xlabel('Importance (Random Forest)')
    ax5.set_title('Top 20 Important Genes')
    ax5.invert_yaxis()

    # ---------- 6. 项目摘要 ----------
    ax6.axis('off')
    ax6.text(0.1, 0.9, "Project Summary (v19_updated_ae)", fontsize=14, fontweight='bold')
    # 从 trad_results 获取 MLP/CNN/AE 指标
    mlp_acc = trad_results['MLP']['accuracy']; mlp_auc = trad_results['MLP']['auc']
    cnn_acc = trad_results['CNN']['accuracy']; cnn_auc = trad_results['CNN']['auc']
    ae_acc = trad_results['AE+LR']['accuracy']; ae_auc = trad_results['AE+LR']['auc']
    ax6.text(0.1, 0.75, f"MLP  Test Acc: {mlp_acc:.4f}, AUC: {mlp_auc:.4f}", fontsize=11)
    ax6.text(0.1, 0.60, f"CNN  Test Acc: {cnn_acc:.4f}, AUC: {cnn_auc:.4f}", fontsize=11)
    ax6.text(0.1, 0.45, f"AE+LR Test Acc: {ae_acc:.4f}, AUC: {ae_auc:.4f}", fontsize=11)
    ax6.text(0.1, 0.30, f"Stacking Acc: {stack_metrics['accuracy']:.4f}, AUC: {stack_metrics['auc']:.4f}", fontsize=11)
    ax6.text(0.1, 0.15, f"Sample sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}", fontsize=11)
    if cv_metrics:
        ax6.text(0.1, 0.05, f"CV (5-fold): Acc={cv_metrics['acc_mean']:.3f}±{cv_metrics['acc_std']:.3f}", fontsize=10)

    plt.tight_layout()
    plt.savefig('final_results_part1.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("第一部分结果图已保存为 final_results_part1.png")
# ==========  Part 2：混淆矩阵与阈值调优 ==========
def plot_final_results_part2(y_test, mlp_metrics, cnn_metrics, ae_metrics, weighted_metrics, stack_metrics,
                             y_proba_mlp, y_proba_cnn, y_proba_ae, weighted_proba, stack_proba,
                             best_threshold_mlp, best_threshold_cnn):
    """
    绘制第二部分：所有混淆矩阵、阈值调优曲线、AE概率分布
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    # 展平
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()

    # 1. MLP混淆矩阵
    cm = confusion_matrix(y_test, mlp_metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'MLP CM (thr={best_threshold_mlp:.3f})')

    # 2. CNN混淆矩阵
    cm_cnn = confusion_matrix(y_test, cnn_metrics['y_pred'])
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title(f'CNN CM (thr={best_threshold_cnn:.3f})')

    # 3. AE+LR混淆矩阵
    cm_ae = confusion_matrix(y_test, ae_metrics['y_pred'])
    sns.heatmap(cm_ae, annot=True, fmt='d', cmap='OrRd', xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('AE+LR CM')

    # 4. 加权平均混淆矩阵
    cm_weighted = confusion_matrix(y_test, weighted_metrics['y_pred'])
    sns.heatmap(cm_weighted, annot=True, fmt='d', cmap='Oranges', xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax4)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    ax4.set_title('Weighted Avg CM')

    # 5. Stacking混淆矩阵
    cm_stack = confusion_matrix(y_test, stack_metrics['y_pred'])
    sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Purples', xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax5)
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('True')
    ax5.set_title('Stacking CM')

    # 6. MLP阈值调优
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_mlp)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    ax6.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    ax6.axvline(x=best_threshold_mlp, color='r', linestyle='--', label=f'Best thr={best_threshold_mlp:.3f}')
    ax6.set_xlabel('Threshold')
    ax6.set_ylabel('F1 Score')
    ax6.set_title('Threshold Tuning (MLP)')
    ax6.legend()

    # 7. CNN阈值调优
    precision_c, recall_c, thresholds_c = precision_recall_curve(y_test, y_proba_cnn)
    f1_scores_c = 2 * (precision_c[:-1] * recall_c[:-1]) / (precision_c[:-1] + recall_c[:-1] + 1e-10)
    ax7.plot(thresholds_c, f1_scores_c, 'g-', label='F1 Score')
    ax7.axvline(x=best_threshold_cnn, color='r', linestyle='--', label=f'Best thr={best_threshold_cnn:.3f}')
    ax7.set_xlabel('Threshold')
    ax7.set_ylabel('F1 Score')
    ax7.set_title('Threshold Tuning (CNN)')
    ax7.legend()

    # 8. AE概率分布
    ax8.hist(y_proba_ae[y_test==0], bins=15, alpha=0.5, label='pCR', color='blue')
    ax8.hist(y_proba_ae[y_test==1], bins=15, alpha=0.5, label='RD', color='red')
    ax8.axvline(0.5, color='k', linestyle='--', label='Threshold=0.5')
    ax8.set_xlabel('Predicted Probability (RD)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('AE+LR Probability Distribution')
    ax8.legend()

    # 9. 额外信息（放模型说明）
    ax9.axis('off')
    ax9.text(0.1, 0.8, "Confusion Matrices Detail", fontsize=14, fontweight='bold')
    ax9.text(0.1, 0.6, "Each CM shows true vs predicted labels", fontsize=12)
    ax9.text(0.1, 0.4, "Thresholds chosen by F1 maximization on validation set", fontsize=12)

    plt.tight_layout()
    plt.savefig('final_results_part2.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("第二部分结果图已保存为 final_results_part2.png")

# ========== 各模型训练时间和预测时间的柱状图==========
def plot_time_comparison(time_info, save_path='time_comparison.png'):
    """
    绘制各模型训练时间和预测时间的柱状图，并在条块上方显示具体数值。
    """
    models = list(time_info.keys())
    fit_times = [time_info[m]['fit_time'] if time_info[m]['fit_time'] is not None else 0 for m in models]
    pred_times = [time_info[m]['predict_time'] if time_info[m]['predict_time'] is not None else 0 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_fit = ax.bar(x - width/2, fit_times, width, label='Training Time', color='skyblue')
    bars_pred = ax.bar(x + width/2, pred_times, width, label='Prediction Time', color='lightcoral')

    # 计算最大时间，用于动态调整数值偏移量和 y 轴上限
    max_time = max(max(fit_times), max(pred_times)) if (fit_times and pred_times) else 1
    offset = max_time * 0.02  # 相对偏移量

    # 在训练时间条块上方添加数值
    for bar, val in zip(bars_fit, fit_times):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 在预测时间条块上方添加数值
    for bar, val in zip(bars_pred, pred_times):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Model Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max_time * 1.2)  # 留出顶部空间

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"时间对比图已保存至 {save_path}")
# ====================================================================
# 12. 交叉验证
# ====================================================================
def cross_val_dl_enhanced(X_raw, y, n_splits=5, pos_weight=1.0, weight_decay=5e-3,
                          n_features_f=1000, n_features_rf=180, dropout_rate=0.5,
                          device='cuda'):
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

        model = EnhancedMLP(input_dim=X_tr_rf.shape[1], hidden_dims=[64,32], dropout_rate=dropout_rate)
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

# ====================================================================
# 13. 主程序 v19_updated_ae （整合所有步骤）
# ====================================================================
def main():
    print("="*60)
    print("步骤1：加载 GSE25066 系列矩阵文件和注释文件")
    expr_df, y, classes, batch_labels = parse_gse25066_series_matrix('GSE25066_series_matrix.txt')
    probe2symbol = parse_gpl96_annotation('GPL96-57554.txt')

    print("\n步骤2：基础预处理（缺失值处理、方差过滤）")
    expr_var_df, var_selector, base_scaler, base_imputer = basic_preprocess(expr_df, variance_thresh=0.5)

    if batch_labels is not None and len(batch_labels) > 0:
        print("\n步骤2b：执行 ComBat 批次校正")
        batch_series = pd.Series(batch_labels).reindex(expr_var_df.columns)
        expr_var_df = combat_correction(expr_var_df, batch_series)
        print(f"批次校正完成，数据形状: {expr_var_df.shape}")
    else:
        print("\n未检测到批次信息，跳过批次校正。")

    X_full = expr_var_df.T.values
    y_full = np.array(y)

    # 划分
    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    print(f"训练+验证集: {X_trainval_raw.shape[0]} 样本, 测试集: {X_test_raw.shape[0]} 样本")

    scaler_final = StandardScaler()
    X_trainval_scaled = scaler_final.fit_transform(X_trainval_raw)
    X_test_scaled = scaler_final.transform(X_test_raw)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval_raw, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    scaler_search = StandardScaler()
    X_train_scaled = scaler_search.fit_transform(X_train_raw)
    X_val_scaled = scaler_search.transform(X_val_raw)

    # F检验和RF选择
    print("\n步骤3：F检验选择1000个特征（仅在训练集上fit）")
    n_features_f = 1000
    selector_f = SelectKBest(f_classif, k=n_features_f)
    X_train_f = selector_f.fit_transform(X_train_scaled, y_train)
    X_val_f = selector_f.transform(X_val_scaled)
    f_mask = selector_f.get_support()
    selected_probes_f = expr_var_df.index[f_mask].tolist()

    print("\n步骤4：RF特征重要性二次筛选（选180个最重要特征，仅在训练集上fit）")
    n_features_rf = 180
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_f, y_train)
    importances = rf_selector.feature_importances_
    rf_idx = np.argsort(importances)[::-1][:n_features_rf]
    X_train_rf = X_train_f[:, rf_idx]
    X_val_rf = X_val_f[:, rf_idx]

    selected_probes = [selected_probes_f[i] for i in rf_idx]
    selected_symbols = [probe2symbol.get(p, p) for p in selected_probes]
    print(f"RF二次筛选后保留 {X_train_rf.shape[1]} 个特征")
    print(f"训练集: {X_train_rf.shape[0]} 样本, 验证集: {X_val_rf.shape[0]} 样本")

    # 超参数搜索（同前）
    print("\n步骤4.5：超参数网格搜索（pos_weight_factor, weight_decay, dropout_rate）")
    search_pos_factors = [0.6, 0.8, 1.0]
    search_wd = [1e-3, 3e-3, 5e-3]
    search_dropout = [0.3, 0.5, 0.7]
    best_score = -1
    best_params = {}

    train_ds_base = TensorDataset(torch.tensor(X_train_rf, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    val_ds_base = TensorDataset(torch.tensor(X_val_rf, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long))

    for pf in search_pos_factors:
        for wd in search_wd:
            for dr in search_dropout:
                class_counts = np.bincount(y_train)
                pos_w = class_counts[0] / class_counts[1] * pf
                print(f"\n尝试 pf={pf}, wd={wd:.1e}, dropout={dr:.1f} -> pos_weight={pos_w:.2f}")

                train_loader = DataLoader(train_ds_base, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_ds_base, batch_size=64, shuffle=False)

                model = EnhancedMLP(input_dim=X_train_rf.shape[1], hidden_dims=[64,32], dropout_rate=dr)
                model, _, _ = train_dl_with_early_stopping(
                    model, train_loader, val_loader, pos_weight=pos_w,
                    epochs=80, patience=10, device=device, weight_decay=wd, lr=0.001, min_delta=1e-4
                )

                logits_val, _ = evaluate_dl_logits(model, val_loader, device)
                T = learn_temperature(logits_val, y_val, init_T=1.0, lr=0.01, epochs=40)

                _, y_proba_val, _ = evaluate_dl(model, val_loader, device, threshold=0.5, temperature=T)
                best_thr = find_optimal_thresholds(y_val, y_proba_val)
                y_pred_val = (y_proba_val >= best_thr).astype(int)
                val_auc = roc_auc_score(y_val, y_proba_val)
                val_f1 = f1_score(y_val, y_pred_val)
                score = (val_auc + val_f1) / 2
                print(f"  -> AUC={val_auc:.4f}, F1={val_f1:.4f}, Score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = {
                        'pos_weight_factor': pf,
                        'weight_decay': wd,
                        'dropout_rate': dr,
                        'pos_weight': pos_w,
                    }

    print(f"\n最佳超参数: {best_params}, 评分={best_score:.4f}")

    pos_weight = best_params['pos_weight']
    weight_decay = best_params['weight_decay']
    dropout_rate = best_params['dropout_rate']

    # 最终训练数据准备
    print("\n步骤5：使用最佳超参数在完整训练集上训练最终模型（无数据泄露）")
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_trainval_raw, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    scaler_final_model = StandardScaler()
    X_train_scaled_final = scaler_final_model.fit_transform(X_train_final)
    X_val_scaled_final = scaler_final_model.transform(X_val_final)
    X_test_scaled_final = scaler_final_model.transform(X_test_raw)

    selector_f_final = SelectKBest(f_classif, k=n_features_f)
    X_train_f_final = selector_f_final.fit_transform(X_train_scaled_final, y_train_final)
    X_val_f_final = selector_f_final.transform(X_val_scaled_final)
    X_test_f_final = selector_f_final.transform(X_test_scaled_final)

    rf_selector_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector_final.fit(X_train_f_final, y_train_final)
    importances_final = rf_selector_final.feature_importances_
    rf_idx_final = np.argsort(importances_final)[::-1][:n_features_rf]
    X_train_rf_final = X_train_f_final[:, rf_idx_final]
    X_val_rf_final = X_val_f_final[:, rf_idx_final]
    X_test_rf_final = X_test_f_final[:, rf_idx_final]

    train_ds = TensorDataset(torch.tensor(X_train_rf_final, dtype=torch.float32),
                             torch.tensor(y_train_final, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_rf_final, dtype=torch.float32),
                           torch.tensor(y_val_final, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_rf_final, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ---- 训练 MLP ----
    print("\n--- 训练 MLP ---")
    start = time.time()
    mlp_model = EnhancedMLP(input_dim=X_train_rf_final.shape[1], hidden_dims=[64,32], dropout_rate=dropout_rate)
    mlp_model, train_losses, val_losses = train_dl_with_early_stopping(
        mlp_model, train_loader, val_loader, pos_weight=pos_weight,
        epochs=200, patience=15, device=device, weight_decay=weight_decay, min_delta=1e-4
    )
    mlp_train_time = time.time() - start
    logits_val_mlp, y_val_np = evaluate_dl_logits(mlp_model, val_loader, device)
    temperature_mlp = learn_temperature(logits_val_mlp, y_val_np, init_T=1.0, lr=0.01, epochs=80)
    _, y_proba_val_mlp_raw, _ = evaluate_dl(mlp_model, val_loader, device, threshold=0.5, temperature=temperature_mlp)
    best_threshold_mlp = find_optimal_thresholds(y_val_final, y_proba_val_mlp_raw)
    y_pred_mlp, y_proba_mlp, _ = evaluate_dl(mlp_model, test_loader, device, threshold=best_threshold_mlp, temperature=temperature_mlp)
    mlp_acc = accuracy_score(y_test, y_pred_mlp)
    mlp_auc = roc_auc_score(y_test, y_proba_mlp)
    mlp_f1 = f1_score(y_test, y_pred_mlp)
    mlp_metrics = {'accuracy': mlp_acc, 'auc': mlp_auc, 'y_pred': y_pred_mlp}
    print(f"\nMLP 测试集 (阈值={best_threshold_mlp:.4f}, T={temperature_mlp:.3f}): Acc={mlp_acc:.4f}, AUC={mlp_auc:.4f}, F1={mlp_f1:.4f}")

    # ---- 训练 CNN ----
    print("\n--- 训练 CNN ---")
    start = time.time()
    cnn_model = EnhancedCNN(input_dim=X_train_rf_final.shape[1], n_conv_layers=2, n_filters=32,
                            kernel_size=5, hidden_dims=[128,64], dropout_rate=dropout_rate)
    cnn_model, _, _ = train_dl_with_early_stopping(
        cnn_model, train_loader, val_loader, pos_weight=pos_weight,
        epochs=200, patience=15, device=device, weight_decay=weight_decay, min_delta=1e-4
    )
    cnn_train_time = time.time() - start
    logits_val_cnn, _ = evaluate_dl_logits(cnn_model, val_loader, device)
    temperature_cnn = learn_temperature(logits_val_cnn, y_val_np, init_T=1.0, lr=0.01, epochs=80)
    _, y_proba_val_cnn_raw, _ = evaluate_dl(cnn_model, val_loader, device, threshold=0.5, temperature=temperature_cnn)
    best_threshold_cnn = find_optimal_thresholds(y_val_final, y_proba_val_cnn_raw)
    y_pred_cnn, y_proba_cnn, _ = evaluate_dl(cnn_model, test_loader, device, threshold=best_threshold_cnn, temperature=temperature_cnn)
    cnn_acc = accuracy_score(y_test, y_pred_cnn)
    cnn_auc = roc_auc_score(y_test, y_proba_cnn)
    cnn_f1 = f1_score(y_test, y_pred_cnn)
    cnn_metrics = {'accuracy': cnn_acc, 'auc': cnn_auc, 'y_pred': y_pred_cnn}
    print(f"CNN 测试集 (阈值={best_threshold_cnn:.4f}, T={temperature_cnn:.3f}): Acc={cnn_acc:.4f}, AUC={cnn_auc:.4f}, F1={cnn_f1:.4f}")

    # ========== 训练自编码器+分类器 ==========
    print("\n--- 训练 Autoencoder + LogisticRegression ---")
    start = time.time()
    ae_model = AutoencoderClassifier(
        encoding_dim=64, 
        hidden_dims=[128, 64],
        ae_epochs=100,
        ae_lr=0.001,
        ae_weight_decay=1e-4,
        classifier=LogisticRegression(class_weight='balanced', C=1.0),
        device=device
    )
    # 注意：自编码器是在训练集上无监督训练，然后提取特征训练分类器
    # 这里传入训练集和标签（标签仅用于分类器）
    ae_model.fit(X_train_rf_final, y_train_final)
    ae_train_time = time.time() - start
    # 评估
    y_proba_ae = ae_model.predict_proba(X_test_rf_final)[:, 1]
    y_pred_ae = ae_model.predict(X_test_rf_final)
    ae_acc = accuracy_score(y_test, y_pred_ae)
    ae_auc = roc_auc_score(y_test, y_proba_ae)
    ae_f1 = f1_score(y_test, y_pred_ae)
    ae_metrics = {'accuracy': ae_acc, 'auc': ae_auc, 'y_pred': y_pred_ae}
    print(f"AE+LR 测试集: Acc={ae_acc:.4f}, AUC={ae_auc:.4f}, F1={ae_f1:.4f}")

    # 交叉验证
    print("\n步骤6：深度学习模型交叉验证 (5-fold，MLP)")
    cv_acc, cv_auc, cv_f1 = cross_val_dl_enhanced(
        X_trainval_raw, y_trainval, n_splits=5,
        pos_weight=pos_weight, weight_decay=weight_decay, device=device,
        n_features_f=n_features_f, n_features_rf=n_features_rf,
        dropout_rate=dropout_rate
    )
    cv_metrics = {
        'acc_mean': np.mean(cv_acc), 'acc_std': np.std(cv_acc),
        'auc_mean': np.mean(cv_auc), 'auc_std': np.std(cv_auc),
        'f1_mean': np.mean(cv_f1), 'f1_std': np.std(cv_f1)
    }

    # 传统模型与集成
    print("\n步骤7：传统机器学习模型与集成（SVM+RF+LDA+MLP+CNN+AE，Stacking启用passthrough）")
    trad_results, weighted_proba, weighted_pred, weighted_scores, \
    stacking_model, stack_proba, stack_pred, stack_scores , time_info= train_traditional_and_ensemble(
        X_train_rf_final, y_train_final, X_val_rf_final, y_val_final, X_test_rf_final, y_test,
        mlp_model, temperature_mlp, best_threshold_mlp,
        cnn_model, temperature_cnn, best_threshold_cnn,
        ae_model
    )
    time_info['MLP']['fit_time'] = mlp_train_time
    time_info['CNN']['fit_time'] = cnn_train_time
    time_info['AE+LR']['fit_time'] = ae_train_time
    weighted_metrics = {'accuracy': weighted_scores[0], 'auc': weighted_scores[1],
                        'y_pred': weighted_pred, 'y_proba': weighted_proba}
    stack_metrics = {'accuracy': stack_scores[0], 'auc': stack_scores[1],
                     'y_pred': stack_pred, 'y_proba': stack_proba}

    # 特征重要性
    rf_model = trad_results['RF']['model']
    importances_rf = rf_model.feature_importances_
    top_n = 20
    sorted_idx = np.argsort(importances_rf)[::-1][:top_n]
    top_symbols = [selected_symbols[i] for i in sorted_idx]
    top_imps = importances_rf[sorted_idx]
    important_info = (top_symbols, top_imps)
    with open('important_genes_final_v19_updated_ae.txt', 'w') as f:
        f.write("Rank\tGene Symbol\tImportance\n")
        for rank, (sym, imp) in enumerate(zip(top_symbols, top_imps), 1):
            f.write(f"{rank}\t{sym}\t{imp:.6f}\n")
    print("重要基因列表已保存到 important_genes_final_v19_updated_ae.txt")

    # 可视化
    print("\n步骤8：生成结果图表")
    plot_final_results_part1(trad_results, mlp_metrics, cnn_metrics, ae_metrics, weighted_metrics, stack_metrics,
                         train_losses, val_losses, y_train_final, y_val_final, y_test,
                         y_proba_mlp, y_proba_cnn, y_proba_ae, weighted_proba, stack_proba,
                         important_info, cv_metrics, best_threshold_mlp, best_threshold_cnn)
    plot_final_results_part2(y_test, mlp_metrics, cnn_metrics, ae_metrics, weighted_metrics, stack_metrics,
                         y_proba_mlp, y_proba_cnn, y_proba_ae, weighted_proba, stack_proba,
                         best_threshold_mlp, best_threshold_cnn)

    # 样本量实验（仅MLP，节省时间）
    print("\n步骤9：样本量 vs 性能学习曲线实验（固定特征数180，MLP）")
    scaler_exp = StandardScaler()
    X_train_raw_scaled = scaler_exp.fit_transform(X_train_raw)
    X_val_raw_scaled = scaler_exp.transform(X_val_raw)
    X_test_raw_scaled = scaler_exp.transform(X_test_raw)

    df_sample = sample_size_experiment(
        X_train_raw_scaled, y_train, X_val_raw_scaled, y_val, X_test_raw_scaled, y_test,
        proportions=np.linspace(0.1, 1.0, 10),
        n_features=n_features_rf,
        device=device,
        pos_weight=pos_weight,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate
    )
    plot_sample_size_curve(df_sample, save_path='sample_size_vs_performance_v19_updated_ae.png')

    # 保存模型
    print("\n步骤10：保存模型和特征选择器")
    save_dir = 'model_artifacts_v19_updated_ae'
    os.makedirs(save_dir, exist_ok=True)

    preprocess_dict = {
        'var_selector': var_selector,
        'base_scaler': base_scaler,
        'base_imputer': base_imputer,
        'scaler_final': scaler_final_model,
        'f_selector': selector_f_final,
        'rf_selector': rf_selector_final,
        'rf_selected_indices': rf_idx_final,
        'selected_probes': selected_probes,
        'selected_symbols': selected_symbols,
        'classes': classes,
        'best_threshold_mlp': best_threshold_mlp,
        'best_threshold_cnn': best_threshold_cnn,
        'pos_weight': pos_weight,
        'temperature_mlp': temperature_mlp,
        'temperature_cnn': temperature_cnn,
        'dropout_rate': dropout_rate,
        'batch_corrected': batch_labels is not None and len(batch_labels)>0,
    }
    joblib.dump(preprocess_dict, os.path.join(save_dir, 'preprocessing.pkl'))

    torch.save({
        'model_state_dict': mlp_model.state_dict(),
        'input_dim': X_train_rf_final.shape[1],
        'hidden_dims': [64, 32],
        'dropout_rate': dropout_rate,
        'best_threshold': best_threshold_mlp,
        'pos_weight': pos_weight,
        'temperature': temperature_mlp,
        'model_type': 'MLP',
    }, os.path.join(save_dir, 'mlp_model.pth'))

    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'input_dim': X_train_rf_final.shape[1],
        'n_conv_layers': 2,
        'n_filters': 32,
        'kernel_size': 5,
        'hidden_dims': [128, 64],
        'dropout_rate': dropout_rate,
        'best_threshold': best_threshold_cnn,
        'pos_weight': pos_weight,
        'temperature': temperature_cnn,
        'model_type': 'CNN',
    }, os.path.join(save_dir, 'cnn_model.pth'))

    # 保存自编码器
    torch.save({
        'autoencoder_state_dict': ae_model.autoencoder.state_dict(),
        'input_dim': X_train_rf_final.shape[1],
        'encoding_dim': ae_model.encoding_dim,
        'hidden_dims': ae_model.hidden_dims,
    }, os.path.join(save_dir, 'autoencoder.pth'))
    # 保存分类器（逻辑回归）
    joblib.dump(ae_model.classifier, os.path.join(save_dir, 'ae_classifier.pkl'))

    traditional_dict = {
        'svm': trad_results['SVM']['model'],
        'rf': trad_results['RF']['model'],
        'lda': trad_results['LDA']['model'],
        'stacking': stacking_model,
    }
    joblib.dump(traditional_dict, os.path.join(save_dir, 'traditional_models.pkl'))

    print(f"所有模型已保存至 '{save_dir}' 目录")

    for name, times in time_info.items():
        print(f"{name}: train={times['fit_time']}, predict={times['predict_time']}")
    plot_time_comparison(time_info)
    print("\n项目执行完毕！所有结果已保存。")

if __name__ == "__main__":
    main()