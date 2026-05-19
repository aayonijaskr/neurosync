"""
neurosync/src/cognitive/model.py
Cognitive state classifier: Focus vs High Mental Effort.

WHY THIS EXISTS:
This is the "brain" of NeuroSync. It takes extracted EEG features and
outputs a probability distribution over cognitive states. We use a small
PyTorch MLP here. The architecture is intentionally lightweight — the demo
should train in seconds, not hours. For the technical depth angle, this module
is where you'd swap in an SNN (snnTorch) in Phase 3.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


LABELS = {0: "focus", 1: "high_effort"}
LABEL_TO_IDX = {v: k for k, v in LABELS.items()}


class CognitiveStateClassifier(nn.Module):
    """
    Lightweight MLP for cognitive state classification from EEG band-power features.

    Architecture rationale:
    - 3 layers is enough to learn the non-linear separability of band-power patterns
    - BatchNorm stabilizes training on small datasets
    - Dropout prevents overfitting on our synthetic data
    """

    def __init__(self, input_dim: int = 40, hidden_dim: int = 64, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CognitiveStateDetector:
    """
    High-level wrapper around the classifier. Handles train/predict/adapt.
    This is the object the rest of the system interacts with.
    """

    def __init__(self, input_dim: int = 40, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CognitiveStateClassifier(input_dim=input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.is_trained = False
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.adaptation_history = []  # for co-adaptation tracking

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> list:
        """
        Train the classifier on extracted EEG features.

        Args:
            X: shape (n_samples, n_features)
            y: shape (n_samples,) — integer labels
        Returns:
            list of training losses per epoch
        """
        # Normalize features
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.feature_mean) / self.feature_std

        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

        self.is_trained = True
        return losses

    def predict(self, features: np.ndarray) -> Tuple[str, float, dict]:
        """
        Predict cognitive state from a feature vector.

        Args:
            features: 1D array of shape (n_features,)

        Returns:
            (state_label, confidence, full_probabilities)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        x_norm = (features - self.feature_mean) / self.feature_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        state = LABELS[pred_idx]
        confidence = float(probs[pred_idx])
        prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(probs))}

        return state, confidence, prob_dict

    def online_adapt(self, features: np.ndarray, true_label: int, lr: float = 1e-4):
        """
        Single-sample online adaptation step — this is the co-adaptation mechanism.

        WHY: When the user implicitly signals that the AI's output wasn't right
        (e.g., re-reads multiple times, requests simplification), we can use that
        as a noisy label to fine-tune the model in real-time.

        Args:
            features: extracted EEG features (1D array)
            true_label: corrected label (0=focus, 1=high_effort)
            lr: adaptation learning rate (small to avoid catastrophic forgetting)
        """
        x_norm = (features - self.feature_mean) / self.feature_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        y_t = torch.tensor([true_label], dtype=torch.long).to(self.device)

        # Temporarily set a smaller LR for online adaptation
        for pg in self.optimizer.param_groups:
            old_lr = pg["lr"]
            pg["lr"] = lr

        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(x_t)
        loss = self.criterion(logits, y_t)
        loss.backward()
        self.optimizer.step()

        for pg in self.optimizer.param_groups:
            pg["lr"] = old_lr

        self.adaptation_history.append({
            "label": LABELS[true_label],
            "loss": float(loss.item()),
        })
        self.model.eval()

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "adaptation_history": self.adaptation_history,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.feature_mean = ckpt["feature_mean"]
        self.feature_std = ckpt["feature_std"]
        self.adaptation_history = ckpt.get("adaptation_history", [])
        self.is_trained = True
