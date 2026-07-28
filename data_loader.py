"""
数据加载与预处理模块
======================
功能：
  - 解析 GSE25066 系列矩阵文件（GEO SOFT 格式）
  - 解析 GPL96 注释文件（探针 → 基因符号映射）
  - 基础预处理（缺失值填充、标准化、方差过滤）
  - ComBat 批次效应校正（可选）
  - 随机森林二次特征选择
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import re

# 尝试导入 pycombat（可选），仅在需要批次校正时使用
try:
    from pycombat import pycombat
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False
    print("pycombat 未安装，批次校正将被跳过。如需使用请运行: pip install pycombat")


# ====================================================================
# 1. 数据加载与注释解析
# ====================================================================

def parse_gse25066_series_matrix(filepath):
    """解析 GSE25066 系列矩阵文件（GEO SOFT 格式）。

    参数
    ----------
    filepath : str
        GEO 系列矩阵文件路径。

    返回
    -------
    df_expr : pd.DataFrame
        表达矩阵，行为探针 ID，列为样本。
    y : np.ndarray
        二分类标签（0 / 1）。
    classes : np.ndarray
        原始类别名称。
    batch_labels : dict 或 None
        样本 ID → 批次的映射。
    """
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
    """解析 GPL96 注释文件，建立探针 ID → 基因符号映射。

    参数
    ----------
    filepath : str
        GPL96 注释文件路径。

    返回
    -------
    probe2symbol : dict
        探针 ID → 基因符号的字典。
    """
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
    """基础预处理流水线：缺失值填充 → 标准化 → 低方差过滤。

    参数
    ----------
    expr_df : pd.DataFrame
        原始表达矩阵（行为探针，列为样本）。
    variance_thresh : float
        方差阈值，低于该值的探针将被移除。

    返回
    -------
    expr_var_df : pd.DataFrame
        预处理后的表达矩阵。
    var_selector : VarianceThreshold
        方差选择器对象。
    scaler : StandardScaler
        标准化器对象。
    imputer : SimpleImputer
        缺失值填充器对象。
    """
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
    """使用 pyCombat 进行批次效应校正。

    参数
    ----------
    expr_df : pd.DataFrame
        表达矩阵（行为探针，列为样本）。
    batch_series : pd.Series
        样本 ID → 批次标签的映射。

    返回
    -------
    pd.DataFrame
        批次校正后的表达矩阵。
    """
    if not COMBAT_AVAILABLE:
        print("pycombat 未安装，跳过批次校正。")
        return expr_df
    samples = expr_df.columns
    batch_list = [batch_series[s] for s in samples]
    le = LabelEncoder()
    batch_encoded = le.fit_transform(batch_list)
    data_corrected = pycombat(expr_df.values, batch_encoded)
    return pd.DataFrame(data_corrected, index=expr_df.index, columns=expr_df.columns)


# ====================================================================
# 4. RF 二次特征选择
# ====================================================================

def select_features_with_rf(X_train, y_train, X_val, X_test=None, n_features=180):
    """使用随机森林重要性进行二次特征选择。

    参数
    ----------
    X_train : np.ndarray
        训练集特征矩阵。
    y_train : np.ndarray
        训练集标签。
    X_val : np.ndarray
        验证集特征矩阵。
    X_test : np.ndarray 或 None
        测试集特征矩阵（可选）。
    n_features : int
        保留的 Top N 特征数。

    返回
    -------
    X_train_sel : np.ndarray
    X_val_sel : np.ndarray
    X_test_sel : np.ndarray 或 None
    sorted_idx : np.ndarray
    rf : RandomForestClassifier
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:n_features]
    X_train_sel = X_train[:, sorted_idx]
    X_val_sel = X_val[:, sorted_idx]
    X_test_sel = X_test[:, sorted_idx] if X_test is not None else None
    return X_train_sel, X_val_sel, X_test_sel, sorted_idx, rf
