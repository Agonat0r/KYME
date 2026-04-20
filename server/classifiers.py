"""
Deep learning classifiers for EMG gesture recognition.

Three classifier backends:
  1. LDA   — sklearn LinearDiscriminantAnalysis (hand-crafted features)
  2. TCN   — Temporal Convolutional Network (raw EMG windows, PyTorch)
  3. Mamba — Simplified State Space Model (raw EMG windows, PyTorch, CPU-safe)

All classifiers implement a common interface:
  .fit(X, y)              — train on windows
  .predict(x) -> (cls, conf)
  .save(path) / .load(path)
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy PyTorch imports (only when TCN/Mamba are actually used) ───────────────

_torch = None
_nn = None
_F = None


def _import_torch():
    global _torch, _nn, _F
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
        _nn = nn
        _F = F
    return _torch, _nn, _F


# =============================================================================
# LDA Classifier (feature-based)
# =============================================================================

class LDAClassifier:
    """LDA on hand-crafted LibEMG features. Fast, needs little data."""

    name = "LDA"

    def __init__(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="svd", tol=1e-4)),
        ])
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Fit on an explicit train/validation split."""
        from sklearn.metrics import accuracy_score

        self._pipe.fit(X_train, y_train)
        self._fitted = True

        val_acc = None
        if X_val is not None and y_val is not None and len(X_val):
            val_acc = float(accuracy_score(y_val, self._pipe.predict(X_val)))
        return {"val_accuracy": val_acc, "n_params": self._count_params()}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """X: (n_samples, n_features), y: (n_samples,)"""
        n = len(X)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        split = min(max(int(n * 0.8), 1), max(n - 1, 1))
        X_tr, X_val = X[idx[:split]], X[idx[split:]]
        y_tr, y_val = y[idx[:split]], y[idx[split:]]
        return self.fit_split(X_tr, y_tr, X_val, y_val)

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """x: (1, n_features) -> (class_idx, confidence)"""
        pred = int(self._pipe.predict(x)[0])
        proba = self._pipe.predict_proba(x)[0]
        return pred, float(proba.max())

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self._pipe.predict(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self._pipe, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self._pipe = pickle.load(f)
        self._fitted = True

    def _count_params(self) -> int:
        lda = self._pipe.named_steps["lda"]
        if hasattr(lda, "coef_"):
            return int(np.prod(lda.coef_.shape) + np.prod(lda.intercept_.shape))
        return 0


# =============================================================================
# TCN Classifier (Temporal Convolutional Network)
# =============================================================================

class TCNClassifier:
    """
    1D Temporal Convolutional Network operating on raw EMG windows.
    Input: (batch, n_channels, seq_len)
    Uses dilated causal convolutions — fast on CPU.
    """

    name = "TCN"

    def __init__(self, n_channels: int = 8, n_classes: int = 5,
                 hidden: int = 64, n_levels: int = 3, kernel_size: int = 3,
                 dropout: float = 0.2, lr: float = 1e-3, epochs: int = 60):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden = hidden
        self.n_levels = n_levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self._model = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _build(self):
        torch, nn, F = _import_torch()

        class CausalConv1d(nn.Module):
            def __init__(self, in_ch, out_ch, k, dilation):
                super().__init__()
                self.pad = (k - 1) * dilation
                self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dilation)

            def forward(self, x):
                x = _F.pad(x, (self.pad, 0))
                return self.conv(x)

        class TCNBlock(nn.Module):
            def __init__(self, in_ch, out_ch, k, dilation, drop):
                super().__init__()
                self.net = nn.Sequential(
                    CausalConv1d(in_ch, out_ch, k, dilation),
                    nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(drop),
                    CausalConv1d(out_ch, out_ch, k, dilation),
                    nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(drop),
                )
                self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.net(x) + self.skip(x))

        class TCNModel(nn.Module):
            def __init__(self, n_ch, n_cls, hid, n_lev, k, drop):
                super().__init__()
                layers = []
                for i in range(n_lev):
                    ic = n_ch if i == 0 else hid
                    layers.append(TCNBlock(ic, hid, k, dilation=2 ** i, drop=drop))
                self.tcn = nn.Sequential(*layers)
                self.head = nn.Linear(hid, n_cls)

            def forward(self, x):
                h = self.tcn(x)
                return self.head(h[:, :, -1])

        self._model = TCNModel(
            self.n_channels, self.n_classes,
            self.hidden, self.n_levels, self.kernel_size, self.dropout,
        )

    def fit_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Fit on an explicit train/validation split."""
        torch, nn, _ = _import_torch()
        self._build()

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32) if X_val is not None and len(X_val) else None
        y_val_t = torch.tensor(y_val, dtype=torch.long) if y_val is not None and len(y_val) else None

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        batch_size = min(64, len(X_tr))

        self._model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), batch_size):
                batch_idx = perm[i:i + batch_size]
                xb, yb = X_tr[batch_idx], y_tr[batch_idx]
                opt.zero_grad()
                loss_fn(self._model(xb), yb).backward()
                opt.step()

        self._model.eval()
        self._fitted = True

        val_acc = None
        if X_val_t is not None and y_val_t is not None and len(X_val_t):
            with torch.no_grad():
                preds = self._model(X_val_t).argmax(dim=1)
                val_acc = float((preds == y_val_t).float().mean())

        return {"val_accuracy": val_acc, "n_params": self._count_params()}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """X: (n_windows, n_channels, seq_len), y: (n_windows,)"""
        n = len(X)
        idx = np.random.default_rng(42).permutation(n)
        split = min(max(int(n * 0.8), 1), max(n - 1, 1))
        return self.fit_split(X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]])

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """x: (1, n_channels, seq_len) raw window"""
        torch, _, _ = _import_torch()
        self._model.eval()
        with torch.no_grad():
            inp = torch.tensor(x, dtype=torch.float32)
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            logits = self._model(inp)
            proba = torch.softmax(logits, dim=1)[0]
            cls = int(proba.argmax())
            conf = float(proba[cls])
        return cls, conf

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        torch, _, _ = _import_torch()
        self._model.eval()
        with torch.no_grad():
            inp = torch.tensor(X, dtype=torch.float32)
            logits = self._model(inp)
            return logits.argmax(dim=1).cpu().numpy()

    def save(self, path: str):
        torch, _, _ = _import_torch()
        torch.save({
            "state_dict": self._model.state_dict(),
            "config": {
                "n_channels": self.n_channels, "n_classes": self.n_classes,
                "hidden": self.hidden, "n_levels": self.n_levels,
                "kernel_size": self.kernel_size, "dropout": self.dropout,
            },
        }, path)

    def load(self, path: str):
        torch, _, _ = _import_torch()
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data["config"]
        self.n_channels = cfg["n_channels"]
        self.n_classes = cfg["n_classes"]
        self.hidden = cfg["hidden"]
        self.n_levels = cfg["n_levels"]
        self.kernel_size = cfg["kernel_size"]
        self.dropout = cfg["dropout"]
        self._build()
        self._model.load_state_dict(data["state_dict"])
        self._model.eval()
        self._fitted = True

    def _count_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters()) if self._model else 0


# =============================================================================
# Mamba / SSM Classifier (State Space Model — CPU-safe, no CUDA required)
# =============================================================================

class MambaClassifier:
    """
    Simplified selective State Space Model inspired by Mamba.
    Pure PyTorch implementation — runs on CPU, no mamba-ssm dependency.
    Input: (batch, n_channels, seq_len)
    """

    name = "Mamba"

    def __init__(self, n_channels: int = 8, n_classes: int = 5,
                 d_model: int = 48, d_state: int = 16, n_layers: int = 2,
                 lr: float = 1e-3, epochs: int = 80):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self._model = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _build(self):
        torch, nn, F = _import_torch()

        class S4Block(nn.Module):
            """Single SSM layer with learnable A, B, C, D matrices."""

            def __init__(self, d, d_state):
                super().__init__()
                self.d = d
                self.d_state = d_state
                # HiPPO-inspired initialization
                self.log_A = nn.Parameter(torch.log(0.5 + torch.rand(d, d_state) * 0.5))
                self.B = nn.Parameter(torch.randn(d, d_state) * 0.02)
                self.C = nn.Parameter(torch.randn(d, d_state) * 0.02)
                self.D = nn.Parameter(torch.ones(d))
                self.log_dt = nn.Parameter(torch.zeros(d) - 1.0)

            def forward(self, x):
                # x: (batch, d, L)
                batch, d, L = x.shape
                dt = _F.softplus(self.log_dt)          # (d,)
                A = -torch.exp(self.log_A)             # (d, s) — stable (negative)
                dA = torch.exp(dt.unsqueeze(-1) * A)   # (d, s) — discretized
                dB = dt.unsqueeze(-1) * self.B         # (d, s)

                h = torch.zeros(batch, d, self.d_state, device=x.device)
                ys = []
                for t in range(L):
                    h = h * dA.unsqueeze(0) + x[:, :, t:t + 1] * dB.unsqueeze(0)
                    y = (h * self.C.unsqueeze(0)).sum(-1) + self.D * x[:, :, t]
                    ys.append(y)
                return torch.stack(ys, dim=-1)  # (batch, d, L)

        class MambaLayer(nn.Module):
            def __init__(self, d, d_state):
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.ssm = S4Block(d, d_state)
                self.gate_proj = nn.Linear(d, d)
                self.out_proj = nn.Linear(d, d)

            def forward(self, x):
                # x: (batch, d, L)
                res = x
                x_t = x.permute(0, 2, 1)  # (batch, L, d)
                x_t = self.norm(x_t)
                x = x_t.permute(0, 2, 1)  # (batch, d, L)
                ssm_out = self.ssm(x)
                gate = torch.sigmoid(self.gate_proj(ssm_out.permute(0, 2, 1)))
                out = self.out_proj(ssm_out.permute(0, 2, 1) * gate)
                return res + out.permute(0, 2, 1)

        class MambaEMG(nn.Module):
            def __init__(self, n_ch, n_cls, d_model, d_state, n_layers):
                super().__init__()
                self.proj_in = nn.Conv1d(n_ch, d_model, 1)
                self.layers = nn.ModuleList([
                    MambaLayer(d_model, d_state) for _ in range(n_layers)
                ])
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, n_cls),
                )

            def forward(self, x):
                x = self.proj_in(x)                  # (batch, d_model, L)
                for layer in self.layers:
                    x = layer(x)
                x = x[:, :, -1]                      # last timestep
                return self.head(x)

        self._model = MambaEMG(
            self.n_channels, self.n_classes,
            self.d_model, self.d_state, self.n_layers,
        )

    def fit_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Fit on an explicit train/validation split."""
        torch, nn, _ = _import_torch()
        self._build()

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32) if X_val is not None and len(X_val) else None
        y_val_t = torch.tensor(y_val, dtype=torch.long) if y_val is not None and len(y_val) else None

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        batch_size = min(32, len(X_tr))

        self._model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), batch_size):
                batch_idx = perm[i:i + batch_size]
                xb, yb = X_tr[batch_idx], y_tr[batch_idx]
                opt.zero_grad()
                loss = loss_fn(self._model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

        self._model.eval()
        self._fitted = True

        val_acc = None
        if X_val_t is not None and y_val_t is not None and len(X_val_t):
            with torch.no_grad():
                preds = self._model(X_val_t).argmax(dim=1)
                val_acc = float((preds == y_val_t).float().mean())

        return {"val_accuracy": val_acc, "n_params": self._count_params()}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """X: (n_windows, n_channels, seq_len), y: (n_windows,)"""
        n = len(X)
        idx = np.random.default_rng(42).permutation(n)
        split = min(max(int(n * 0.8), 1), max(n - 1, 1))
        return self.fit_split(X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]])

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """x: (1, n_channels, seq_len) or (n_channels, seq_len)"""
        torch, _, _ = _import_torch()
        self._model.eval()
        with torch.no_grad():
            inp = torch.tensor(x, dtype=torch.float32)
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            logits = self._model(inp)
            proba = torch.softmax(logits, dim=1)[0]
            cls = int(proba.argmax())
            conf = float(proba[cls])
        return cls, conf

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        torch, _, _ = _import_torch()
        self._model.eval()
        with torch.no_grad():
            inp = torch.tensor(X, dtype=torch.float32)
            logits = self._model(inp)
            return logits.argmax(dim=1).cpu().numpy()

    def save(self, path: str):
        torch, _, _ = _import_torch()
        torch.save({
            "state_dict": self._model.state_dict(),
            "config": {
                "n_channels": self.n_channels, "n_classes": self.n_classes,
                "d_model": self.d_model, "d_state": self.d_state,
                "n_layers": self.n_layers,
            },
        }, path)

    def load(self, path: str):
        torch, _, _ = _import_torch()
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data["config"]
        self.n_channels = cfg["n_channels"]
        self.n_classes = cfg["n_classes"]
        self.d_model = cfg["d_model"]
        self.d_state = cfg["d_state"]
        self.n_layers = cfg["n_layers"]
        self._build()
        self._model.load_state_dict(data["state_dict"])
        self._model.eval()
        self._fitted = True

    def _count_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters()) if self._model else 0


# =============================================================================
# Classifier registry
# =============================================================================

CLASSIFIERS = {
    "LDA":   LDAClassifier,
    "TCN":   TCNClassifier,
    "Mamba": MambaClassifier,
}


def get_classifier(name: str, **kwargs):
    cls = CLASSIFIERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown classifier '{name}'. Options: {list(CLASSIFIERS.keys())}")
    return cls(**kwargs)
