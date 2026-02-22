"""
LSTM model for sequential market downturn prediction.

Implements a PyTorch-based LSTM that processes 30-day sequences of
technical and sentiment features to predict upcoming market drops.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM neural network for time-series classification.

    Architecture: LSTM(hidden=64, layers=2, dropout=0.3) -> Linear -> Sigmoid.
    Trained with Adam optimizer and BCELoss with early stopping.

    Supports creating sequence inputs from flat feature matrices
    by grouping consecutive time steps.

    Attributes:
        sequence_length: Number of time steps per input sequence.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers.
        learning_rate: Adam optimizer learning rate.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without improvement).
        random_state: Random seed for reproducibility.
        model: The PyTorch LSTM model.
        scaler: StandardScaler for feature normalization.
        name: Human-readable model name.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 7,
        random_state: int = 42,
    ) -> None:
        """Initialize the LSTM model.

        Args:
            sequence_length: Days in each input sequence (default 30).
            hidden_size: LSTM hidden state dimension (default 64).
            num_layers: Number of stacked LSTM layers (default 2).
            dropout: Dropout between LSTM layers (default 0.3).
            learning_rate: Adam learning rate (default 0.001).
            batch_size: Training mini-batch size (default 64).
            epochs: Maximum training epochs (default 50).
            patience: Early stopping patience (default 7).
            random_state: Random seed for reproducibility.
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.name = "LSTM"

        self.model = None
        self.scaler = None
        self._device = None
        self._input_size: Optional[int] = None

    def _build_model(self, input_size: int) -> None:
        """Build the PyTorch LSTM model.

        Args:
            input_size: Number of input features.
        """
        import torch
        import torch.nn as nn

        self._input_size = input_size

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _LSTMNet(nn.Module):
            """Inner LSTM network class."""

            def __init__(
                self_inner,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                dropout: float,
            ) -> None:
                super().__init__()
                self_inner.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self_inner.fc = nn.Linear(hidden_size, 1)
                self_inner.sigmoid = nn.Sigmoid()

            def forward(self_inner, x: "torch.Tensor") -> "torch.Tensor":
                lstm_out, _ = self_inner.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                output = self_inner.sigmoid(self_inner.fc(last_hidden))
                return output.squeeze(-1)

        self.model = _LSTMNet(
            input_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self._device)

        logger.info(
            "Built LSTM: input=%d, hidden=%d, layers=%d, dropout=%.1f, device=%s",
            input_size, self.hidden_size, self.num_layers, self.dropout, self._device,
        )

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create overlapping sequences from flat feature data.

        Groups consecutive rows into sequences of length `sequence_length`.
        The target for each sequence is the target of the last row.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).

        Returns:
            Tuple of (X_seq, y_seq) where X_seq has shape
            (n_sequences, sequence_length, n_features) and y_seq has
            shape (n_sequences,).
        """
        sequences: List[np.ndarray] = []
        targets: List[float] = []

        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i - self.sequence_length: i])
            targets.append(y[i])

        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(targets, dtype=np.float32)

        logger.info(
            "Created %d sequences of length %d (from %d samples)",
            len(X_seq), self.sequence_length, len(X),
        )
        return X_seq, y_seq

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMModel":
        """Train the LSTM model with early stopping.

        If validation data is provided, monitors validation loss for
        early stopping. Otherwise, uses a portion of training data.

        Args:
            X: Training feature matrix (n_samples, n_features) or
               pre-sequenced (n_sequences, seq_len, n_features).
            y: Training target array.
            X_val: Optional validation features (same format as X).
            y_val: Optional validation targets.

        Returns:
            self, for method chaining.
        """
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        # Initialize scaler and scale data
        self.scaler = StandardScaler()

        # Check if data is already sequenced
        if X.ndim == 2:
            X_scaled = self.scaler.fit_transform(X).astype(np.float32)
            X_seq, y_seq = self.create_sequences(X_scaled, y)
        else:
            # Already sequenced - scale each feature across all timesteps
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled_flat = self.scaler.fit_transform(X_flat).astype(np.float32)
            X_seq = X_scaled_flat.reshape(original_shape)
            y_seq = y

        if X_val is not None and y_val is not None:
            if X_val.ndim == 2:
                X_val_scaled = self.scaler.transform(X_val).astype(np.float32)
                X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val)
            else:
                original_shape = X_val.shape
                X_val_flat = X_val.reshape(-1, X_val.shape[-1])
                X_val_scaled = self.scaler.transform(X_val_flat).astype(np.float32)
                X_val_seq = X_val_scaled.reshape(original_shape)
                y_val_seq = y_val
        else:
            X_val_seq, y_val_seq = None, None

        if len(X_seq) == 0:
            logger.warning("No sequences created. Insufficient data for LSTM.")
            return self

        input_size = X_seq.shape[2]
        self._build_model(input_size)

        # Convert to tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self._device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self._device)

        if X_val_seq is not None and len(X_val_seq) > 0:
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(self._device)
            y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).to(self._device)
        else:
            X_val_tensor, y_val_tensor = None, None

        # Loss and optimizer
        # Use class weights for imbalanced data
        pos_weight = torch.tensor(
            [(len(y_seq) - y_seq.sum()) / max(y_seq.sum(), 1)],
            dtype=torch.float32,
        ).to(self._device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        logger.info(
            "Training LSTM: %d train sequences, %s val sequences, max %d epochs",
            len(X_seq),
            len(X_val_seq) if X_val_seq is not None else "no",
            self.epochs,
        )

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,  # No shuffle for time series
        )

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_val_tensor)
                    val_loss = criterion(val_output, y_val_tensor).item()

                if (epoch + 1) % 5 == 0:
                    logger.info(
                        "  Epoch %d/%d - train_loss: %.4f, val_loss: %.4f",
                        epoch + 1, self.epochs, avg_train_loss, val_loss,
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            "  Early stopping at epoch %d (val_loss=%.4f, best=%.4f)",
                            epoch + 1, val_loss, best_val_loss,
                        )
                        break
            else:
                if (epoch + 1) % 5 == 0:
                    logger.info(
                        "  Epoch %d/%d - train_loss: %.4f",
                        epoch + 1, self.epochs, avg_train_loss,
                    )

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best model (val_loss=%.4f)", best_val_loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features) or
               pre-sequenced (n_sequences, seq_len, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions.

        Args:
            X: Feature matrix (n_samples, n_features) or
               pre-sequenced (n_sequences, seq_len, n_features).

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if X.ndim == 2:
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X).astype(np.float32)
            else:
                X_scaled = X.astype(np.float32)
            # Create sequences - use only complete sequences
            sequences = []
            for i in range(self.sequence_length, len(X_scaled)):
                sequences.append(X_scaled[i - self.sequence_length: i])
            if not sequences:
                return np.array([[0.5, 0.5]])
            X_input = np.array(sequences, dtype=np.float32)
        else:
            if self.scaler is not None:
                original_shape = X.shape
                X_flat = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.transform(X_flat).astype(np.float32)
                X_input = X_scaled.reshape(original_shape)
            else:
                X_input = X.astype(np.float32)

        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(self._device)

        self.model.eval()
        with torch.no_grad():
            proba_pos = self.model(X_tensor).cpu().numpy()

        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model and return metrics.

        Handles the sequence-creation offset when computing metrics.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: True labels.

        Returns:
            Dictionary containing accuracy, precision, recall, F1, ROC-AUC.
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_proba = self.predict_proba(X)

        # Adjust y to match sequence output (drop first sequence_length items)
        if X.ndim == 2:
            y_aligned = y[self.sequence_length:]
        else:
            y_aligned = y

        # Ensure lengths match
        min_len = min(len(y_aligned), len(y_proba))
        y_aligned = y_aligned[:min_len]
        y_proba = y_proba[:min_len]
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)

        metrics: Dict[str, Any] = {
            "model": self.name,
            "accuracy": float(accuracy_score(y_aligned, y_pred)),
            "precision": float(precision_score(y_aligned, y_pred, zero_division=0)),
            "recall": float(recall_score(y_aligned, y_pred, zero_division=0)),
            "f1": float(f1_score(y_aligned, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_aligned, y_proba[:, 1])),
        }

        logger.info(
            "%s evaluation -> Accuracy: %.3f, F1: %.3f, ROC-AUC: %.3f",
            self.name, metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        )
        logger.info("\n%s", classification_report(y_aligned, y_pred, zero_division=0))

        return metrics

    def save(self, path: str) -> None:
        """Save model state and scaler to disk.

        Args:
            path: File path for saving (e.g. 'models/lstm.pt').
        """
        import torch
        import joblib

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_size": self._input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
        }, output_path)

        # Save scaler alongside
        scaler_path = output_path.with_suffix(".scaler.pkl")
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)

        logger.info("Saved %s to %s", self.name, output_path)

    def load(self, path: str) -> "LSTMModel":
        """Load a previously saved model.

        Args:
            path: File path to the saved model.

        Returns:
            self, with the loaded model.
        """
        import torch
        import joblib

        checkpoint = torch.load(path, map_location="cpu")
        self._input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.sequence_length = checkpoint["sequence_length"]

        self._build_model(self._input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        scaler_path = Path(path).with_suffix(".scaler.pkl")
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        logger.info("Loaded %s from %s", self.name, path)
        return self
