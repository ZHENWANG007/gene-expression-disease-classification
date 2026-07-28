"""
主程序入口
==========
完整的基因表达谱疾病分类流水线：
  数据加载 → 预处理 → 特征选择 → 深度学习训练 → 集成 → 评估可视化 → 模型保存
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
import joblib
import os
import warnings

# 项目模块
from data_loader import (parse_gse25066_series_matrix, parse_gpl96_annotation,
                         basic_preprocess, combat_correction)
from model import EnhancedMLP, EnhancedCNN, AutoencoderClassifier
from train import (train_dl_with_early_stopping, learn_temperature,
                   find_optimal_thresholds, train_traditional_and_ensemble,
                   sample_size_experiment, cross_val_dl_enhanced)
from evaluate import (evaluate_dl_logits, evaluate_dl,
                      plot_sample_size_curve, plot_final_results_part1,
                      plot_final_results_part2, plot_time_comparison)

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def main():
    print("=" * 60)
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

    # F 检验 + RF 选择
    print("\n步骤3：F 检验选择 1000 个特征（仅在训练集上 fit）")
    n_features_f = 1000
    selector_f = SelectKBest(f_classif, k=n_features_f)
    X_train_f = selector_f.fit_transform(X_train_scaled, y_train)
    X_val_f = selector_f.transform(X_val_scaled)
    f_mask = selector_f.get_support()
    selected_probes_f = expr_var_df.index[f_mask].tolist()

    print("\n步骤4：RF 特征重要性二次筛选（选 180 个最重要特征，仅在训练集上 fit）")
    n_features_rf = 180
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_f, y_train)
    importances = rf_selector.feature_importances_
    rf_idx = np.argsort(importances)[::-1][:n_features_rf]
    X_train_rf = X_train_f[:, rf_idx]
    X_val_rf = X_val_f[:, rf_idx]

    selected_probes = [selected_probes_f[i] for i in rf_idx]
    selected_symbols = [probe2symbol.get(p, p) for p in selected_probes]
    print(f"RF 二次筛选后保留 {X_train_rf.shape[1]} 个特征")
    print(f"训练集: {X_train_rf.shape[0]} 样本, 验证集: {X_val_rf.shape[0]} 样本")

    # 超参数搜索
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

                model = EnhancedMLP(input_dim=X_train_rf.shape[1], hidden_dims=[64, 32], dropout_rate=dr)
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
    mlp_model = EnhancedMLP(input_dim=X_train_rf_final.shape[1], hidden_dims=[64, 32], dropout_rate=dropout_rate)
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
                            kernel_size=5, hidden_dims=[128, 64], dropout_rate=dropout_rate)
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

    # ---- 训练自编码器 + 分类器 ----
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
    ae_model.fit(X_train_rf_final, y_train_final)
    ae_train_time = time.time() - start
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
    print("\n步骤7：传统机器学习模型与集成（SVM+RF+LDA+MLP+CNN+AE，Stacking 启用 passthrough）")
    trad_results, weighted_proba, weighted_pred, weighted_scores, \
    stacking_model, stack_proba, stack_pred, stack_scores, time_info = train_traditional_and_ensemble(
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
    plot_final_results_part1(trad_results, mlp_metrics, cnn_metrics, ae_metrics,
                             weighted_metrics, stack_metrics,
                             train_losses, val_losses, y_train_final, y_val_final, y_test,
                             y_proba_mlp, y_proba_cnn, y_proba_ae,
                             weighted_proba, stack_proba,
                             important_info, cv_metrics, best_threshold_mlp, best_threshold_cnn)
    plot_final_results_part2(y_test, mlp_metrics, cnn_metrics, ae_metrics,
                             weighted_metrics, stack_metrics,
                             y_proba_mlp, y_proba_cnn, y_proba_ae,
                             weighted_proba, stack_proba,
                             best_threshold_mlp, best_threshold_cnn)

    # 样本量实验
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
        'batch_corrected': batch_labels is not None and len(batch_labels) > 0,
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
