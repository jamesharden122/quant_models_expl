from __future__ import annotations

from pathlib import Path
from typing import Dict
from datetime import datetime

import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from base_trainer import BaseTrainer, TrainingResult


def _r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


class MLSLSTMTrainer(BaseTrainer):
    """Multi-Layer Sequential LSTM trainer for single-asset forecasting."""

    def __init__(self, csv_path: str, time_step: int = 100,
                 epochs: int = 100, batch_size: int = 64,
                 validation_split: float = 0.1) -> None:
        self.csv_path = csv_path
        self.time_step = time_step
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.min: float | None = None
        self.max: float | None = None

    # --------------------------- helpers ---------------------------
    def _load_series(self) -> np.ndarray:
        df = pl.read_csv(self.csv_path, columns=["Date", "Close"], try_parse_dates=True)
        df = df.sort("Date").fill_null(strategy="forward")
        return df["Close"].to_numpy()

    def _scale_train_test(self, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.min = float(train.min())
        self.max = float(train.max())
        scale = lambda x: (x - self.min) / (self.max - self.min)
        return scale(train), scale(test)

    def _inverse(self, data: np.ndarray) -> np.ndarray:
        assert self.min is not None and self.max is not None
        return data * (self.max - self.min) + self.min

    def _create_dataset(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(series) - self.time_step - 1):
            X.append(series[i:(i + self.time_step)])
            y.append(series[i + self.time_step])
        return np.array(X), np.array(y)

    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.time_step, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(1),
        ])
        model.compile(loss="mse", optimizer="adam", metrics=[_r2_metric])
        return model

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert self.max is not None and self.min is not None
        nrmse = rmse / (self.max - self.min)
        return {"r2": float(r2), "mape": float(mape), "nrmse": float(nrmse)}

    # ------------------------- main train -------------------------
    def train(self) -> TrainingResult:
        series = self._load_series()
        train_size = int(len(series) * 0.65)
        train_series, test_series = series[:train_size], series[train_size:]
        train_scaled, test_scaled = self._scale_train_test(train_series, test_series)

        X_train, y_train = self._create_dataset(train_scaled)
        X_test, y_test = self._create_dataset(test_scaled)
        X_train = X_train.reshape(X_train.shape[0], self.time_step, 1)
        X_test = X_test.reshape(X_test.shape[0], self.time_step, 1)

        model = self._build_model()
        model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=0,
        )

        train_pred = model.predict(X_train, verbose=0).squeeze()
        test_pred = model.predict(X_test, verbose=0).squeeze()
        train_pred_inv = self._inverse(train_pred)
        test_pred_inv = self._inverse(test_pred)
        y_train_inv = self._inverse(y_train)
        y_test_inv = self._inverse(y_test)

        metrics = {
            "train": self._compute_metrics(y_train_inv, train_pred_inv),
            "test": self._compute_metrics(y_test_inv, test_pred_inv),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(__file__).parent / "saved" / f"mls_lstm_{timestamp}"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_dir, include_optimizer=False)

        return TrainingResult(model_dir=str(save_dir), metrics=metrics)
