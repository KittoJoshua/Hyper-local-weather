"""
Long Short-Term Memory (LSTM) Deep Learning Model for Temporal Weather Sequences.
Captures seasonal auto-correlation and multi-day climatic cycles using PyTorch.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Tuple, List
from ml_engine.config import LSTM_PARAMS, WEIGHTS_DIR, TARGET_VARIABLES


class LSTMNetwork(nn.Module):
    """
    PyTorch Recurrent Architecture featuring stacked LSTM layers,
    dropout regularization, and a dense multi-variable output layer.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        # Use last hidden state
        last_out = out[:, -1, :]
        preds = self.fc(last_out)
        return preds


class MicroClimateLSTM:
    """
    Wrapper for training, evaluating, and running inference with the PyTorch LSTM network.
    """

    def __init__(self, input_size: int = 7, output_size: int = 3, params: Dict = None):
        self.input_size = input_size
        self.output_size = output_size
        self.params = params or LSTM_PARAMS.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.params.get("hidden_size", 64),
            num_layers=self.params.get("num_layers", 2),
            output_size=output_size,
            dropout=self.params.get("dropout", 0.2)
        ).to(self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Trains the LSTM network using Adam optimizer and Mean Squared Error loss.
        """
        batch_size = self.params.get("batch_size", 32)
        epochs = self.params.get("epochs", 25)
        lr = self.params.get("learning_rate", 0.001)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.network.train()
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.network(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item() * len(batch_x)

            epoch_train_loss = total_train_loss / len(X_train)
            history["train_loss"].append(epoch_train_loss)

            if X_val is not None and y_val is not None:
                self.network.eval()
                with torch.no_grad():
                    val_x = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    val_y = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    val_preds = self.network(val_x)
                    val_loss = criterion(val_preds, val_y).item()
                    history["val_loss"].append(val_loss)
                self.network.train()

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates continuous predictions for input sequence batch.
        """
        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.network(x_tensor)
            return preds.cpu().numpy()

    def save(self, filepath: str = None):
        path = filepath or str(WEIGHTS_DIR / "lstm_weights.pt")
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "params": self.params
        }, path)

    def load(self, filepath: str = None):
        path = filepath or str(WEIGHTS_DIR / "lstm_weights.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.input_size = checkpoint["input_size"]
            self.output_size = checkpoint["output_size"]
            self.params = checkpoint.get("params", self.params)
            self.network = LSTMNetwork(
                input_size=self.input_size,
                hidden_size=self.params.get("hidden_size", 64),
                num_layers=self.params.get("num_layers", 2),
                output_size=self.output_size,
                dropout=self.params.get("dropout", 0.2)
            ).to(self.device)
            self.network.load_state_dict(checkpoint["model_state_dict"])
            self.network.eval()
        return self
