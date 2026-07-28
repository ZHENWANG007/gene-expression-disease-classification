"""
评估与可视化模块
================
功能：
  - 模型评估（输出 logits / 概率 / 预测）
  - ROC 曲线、性能对比条形图、混淆矩阵
  - 学习曲线、样本量实验曲线
  - 各模型训练 / 预测时间对比
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             confusion_matrix, f1_score, recall_score)


# ====================================================================
# 1. 模型评估
# ====================================================================

def evaluate_dl_logits(model, loader, device='cuda'):
    """获取模型在所有样本上的原始 logits。

    返回
    -------
    logits : np.ndarray
    labels : np.ndarray
    """
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
    """完整评估：输出预测类别、概率和真实标签。

    返回
    -------
    preds : np.ndarray
    probs : np.ndarray
    labels : np.ndarray
    """
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


# ====================================================================
# 2. 可视化
# ====================================================================

# ----- 样本量实验曲线 -----
def plot_sample_size_curve(df_results, save_path='sample_size_vs_performance_v19_updated_ae.png'):
    """绘制样本量 vs 性能（Accuracy / AUC）曲线。"""
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


# ----- Part 1：ROC / 学习曲线 / 性能对比 / 重要基因 / 项目摘要 -----
def plot_final_results_part1(trad_results, mlp_metrics, cnn_metrics, ae_metrics,
                             weighted_metrics, stack_metrics,
                             train_losses, val_losses, y_train, y_val, y_test,
                             y_proba_mlp, y_proba_cnn, y_proba_ae,
                             weighted_proba, stack_proba,
                             important_info, cv_metrics,
                             best_threshold_mlp, best_threshold_cnn):
    """绘制第一部分结果图：ROC、学习曲线、性能条形图、F1/Recall、重要基因、摘要。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # ---------- 1. ROC 曲线 ----------
    for name, res in trad_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", lw=1.5)
    fpr, tpr, _ = roc_curve(y_test, weighted_proba)
    ax1.plot(fpr, tpr, label=f"Weighted Avg (AUC={weighted_metrics['auc']:.3f})", lw=2, color='gold')
    fpr, tpr, _ = roc_curve(y_test, stack_proba)
    ax1.plot(fpr, tpr, label=f"Stacking (AUC={stack_metrics['auc']:.3f})", lw=2, color='magenta')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
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

    # ---------- 3. 性能对比条形图 ----------
    base_names = list(trad_results.keys())
    base_accs = [res['accuracy'] for res in trad_results.values()]
    base_aucs = [res['auc'] for res in trad_results.values()]
    ensemble_names = ['WeightedAvg', 'Stacking']
    ensemble_accs = [weighted_metrics['accuracy'], stack_metrics['accuracy']]
    ensemble_aucs = [weighted_metrics['auc'], stack_metrics['auc']]

    names = base_names + ensemble_names
    accs = base_accs + ensemble_accs
    aucs = base_aucs + ensemble_aucs

    x = np.arange(len(names))
    width = 0.35
    bars_acc = ax3.bar(x - width / 2, accs, width, label='Accuracy', color='skyblue')
    bars_auc = ax3.bar(x + width / 2, aucs, width, label='AUC', color='lightcoral')
    for bar, val in zip(bars_acc, accs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars_auc, aucs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15, fontsize=8)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance (Accuracy & AUC)')
    ax3.legend()

    # ---------- 4. F1 分数 + 召回率 ----------
    base_preds = [res['y_pred'] for res in trad_results.values()]
    ensemble_preds = [weighted_metrics['y_pred'], stack_metrics['y_pred']]
    all_preds = base_preds + ensemble_preds
    all_names = base_names + ensemble_names

    f1_scores = [f1_score(y_test, pred) for pred in all_preds]
    recall_scores = [recall_score(y_test, pred) for pred in all_preds]
    x2 = np.arange(len(all_names))
    width2 = 0.35
    bars_f1 = ax4.bar(x2 - width2 / 2, f1_scores, width2, label='F1 Score', color='mediumseagreen')
    bars_rec = ax4.bar(x2 + width2 / 2, recall_scores, width2, label='Recall', color='orange')
    for bar, val in zip(bars_f1, f1_scores):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars_rec, recall_scores):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
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


# ----- Part 2：混淆矩阵 + 阈值调优 + 概率分布 -----
def plot_final_results_part2(y_test, mlp_metrics, cnn_metrics, ae_metrics,
                             weighted_metrics, stack_metrics,
                             y_proba_mlp, y_proba_cnn, y_proba_ae,
                             weighted_proba, stack_proba,
                             best_threshold_mlp, best_threshold_cnn):
    """绘制第二部分结果图：混淆矩阵、阈值调优、AE 概率分布。"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()

    # 1. MLP 混淆矩阵
    cm = confusion_matrix(y_test, mlp_metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax1)
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
    ax1.set_title(f'MLP CM (thr={best_threshold_mlp:.3f})')

    # 2. CNN 混淆矩阵
    cm_cnn = confusion_matrix(y_test, cnn_metrics['y_pred'])
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens',
                xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax2)
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('True')
    ax2.set_title(f'CNN CM (thr={best_threshold_cnn:.3f})')

    # 3. AE+LR 混淆矩阵
    cm_ae = confusion_matrix(y_test, ae_metrics['y_pred'])
    sns.heatmap(cm_ae, annot=True, fmt='d', cmap='OrRd',
                xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax3)
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('True')
    ax3.set_title('AE+LR CM')

    # 4. 加权平均混淆矩阵
    cm_weighted = confusion_matrix(y_test, weighted_metrics['y_pred'])
    sns.heatmap(cm_weighted, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax4)
    ax4.set_xlabel('Predicted'); ax4.set_ylabel('True')
    ax4.set_title('Weighted Avg CM')

    # 5. Stacking 混淆矩阵
    cm_stack = confusion_matrix(y_test, stack_metrics['y_pred'])
    sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Purples',
                xticklabels=['pCR', 'RD'], yticklabels=['pCR', 'RD'], ax=ax5)
    ax5.set_xlabel('Predicted'); ax5.set_ylabel('True')
    ax5.set_title('Stacking CM')

    # 6. MLP 阈值调优
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_mlp)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    ax6.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    ax6.axvline(x=best_threshold_mlp, color='r', linestyle='--', label=f'Best thr={best_threshold_mlp:.3f}')
    ax6.set_xlabel('Threshold'); ax6.set_ylabel('F1 Score')
    ax6.set_title('Threshold Tuning (MLP)')
    ax6.legend()

    # 7. CNN 阈值调优
    precision_c, recall_c, thresholds_c = precision_recall_curve(y_test, y_proba_cnn)
    f1_scores_c = 2 * (precision_c[:-1] * recall_c[:-1]) / (precision_c[:-1] + recall_c[:-1] + 1e-10)
    ax7.plot(thresholds_c, f1_scores_c, 'g-', label='F1 Score')
    ax7.axvline(x=best_threshold_cnn, color='r', linestyle='--', label=f'Best thr={best_threshold_cnn:.3f}')
    ax7.set_xlabel('Threshold'); ax7.set_ylabel('F1 Score')
    ax7.set_title('Threshold Tuning (CNN)')
    ax7.legend()

    # 8. AE 概率分布
    ax8.hist(y_proba_ae[y_test == 0], bins=15, alpha=0.5, label='pCR', color='blue')
    ax8.hist(y_proba_ae[y_test == 1], bins=15, alpha=0.5, label='RD', color='red')
    ax8.axvline(0.5, color='k', linestyle='--', label='Threshold=0.5')
    ax8.set_xlabel('Predicted Probability (RD)'); ax8.set_ylabel('Frequency')
    ax8.set_title('AE+LR Probability Distribution')
    ax8.legend()

    # 9. 图例说明
    ax9.axis('off')
    ax9.text(0.1, 0.8, "Confusion Matrices Detail", fontsize=14, fontweight='bold')
    ax9.text(0.1, 0.6, "Each CM shows true vs predicted labels", fontsize=12)
    ax9.text(0.1, 0.4, "Thresholds chosen by F1 maximization on validation set", fontsize=12)

    plt.tight_layout()
    plt.savefig('final_results_part2.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("第二部分结果图已保存为 final_results_part2.png")


# ----- 训练 / 预测时间对比 -----
def plot_time_comparison(time_info, save_path='time_comparison.png'):
    """绘制各模型训练时间 / 预测时间的柱状图。"""
    models = list(time_info.keys())
    fit_times = [time_info[m]['fit_time'] if time_info[m]['fit_time'] is not None else 0 for m in models]
    pred_times = [time_info[m]['predict_time'] if time_info[m]['predict_time'] is not None else 0 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_fit = ax.bar(x - width / 2, fit_times, width, label='Training Time', color='skyblue')
    bars_pred = ax.bar(x + width / 2, pred_times, width, label='Prediction Time', color='lightcoral')

    max_time = max(max(fit_times), max(pred_times)) if (fit_times and pred_times) else 1
    offset = max_time * 0.02

    for bar, val in zip(bars_fit, fit_times):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    for bar, val in zip(bars_pred, pred_times):
        if val > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model'); ax.set_ylabel('Time (seconds)')
    ax.set_title('Model Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max_time * 1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"时间对比图已保存至 {save_path}")
