"""
模型定义模块
============
包含所有深度学习模型及 sklearn 包装器：
  - EnhancedMLP     : 多层感知机分类器
  - EnhancedCNN     : 一维卷积神经网络分类器
  - Autoencoder     : 自编码器（无监督特征提取）
  - MLPWrapper      : sklearn 兼容包装器（用于 MLP / CNN）
  - AutoencoderClassifier : 自编码器 + 逻辑回归分类器
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ====================================================================
# 1. 深度学习模型
# ====================================================================

class EnhancedMLP(nn.Module):
    """带 BatchNorm + Dropout 的多层感知机二分类器。"""

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
    """带 BatchNorm + Dropout 的一维卷积神经网络二分类器。"""

    def __init__(self, input_dim, n_conv_layers=2, n_filters=32, kernel_size=5,
                 hidden_dims=[128, 64], dropout_rate=0.5):
        super(EnhancedCNN, self).__init__()
        conv_layers = []
        in_channels = 1
        for i in range(n_conv_layers):
            out_channels = n_filters * (2 ** i)
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
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


# ====================================================================
# 2. 自编码器
# ====================================================================

class Autoencoder(nn.Module):
    """自编码器：编码器压缩输入到低维空间，解码器重建。"""

    def __init__(self, input_dim, encoding_dim=64, hidden_dims=[128, 64]):
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
# 3. sklearn 兼容包装器
# ====================================================================

class MLPWrapper(BaseEstimator, ClassifierMixin):
    """将 PyTorch 模型（MLP / CNN）包装为 sklearn 兼容接口。"""

    def __init__(self, model, temperature=1.0, threshold=0.5, device='cuda'):
        self.model = model
        self.temperature = temperature
        self.threshold = threshold
        self.device = device
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("MLPWrapper 仅支持二分类")
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


class AutoencoderClassifier(BaseEstimator, ClassifierMixin):
    """自编码器 + 下游分类器（默认逻辑回归）的端到端模型。"""

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

        # 使用独立训练函数训练自编码器（避免循环导入）
        from train import train_autoencoder
        ae = train_autoencoder(ae, train_loader, epochs=self.ae_epochs,
                               lr=self.ae_lr, weight_decay=self.ae_weight_decay,
                               device=self.device)
        self.autoencoder = ae

        with torch.no_grad():
            X_encoded = ae.encode(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        self.classifier.fit(X_encoded, y)
        self.classes_ = np.unique(y)
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
