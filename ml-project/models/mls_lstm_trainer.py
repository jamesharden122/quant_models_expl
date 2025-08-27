from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
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
    """
    Multi-Layer Sequential LSTM trainer for single-asset forecasting.

    New: supports training from TFRecord files via tf.data:
      - Provide tfrecord_paths=[...] (train) and optionally test_tfrecord_paths=[...]
      - feature_key identifies the scalar field (e.g., "Close" or "value")
      - gzip=True if files are .tfrecord.gz

    The previous CSV mode is removed in this variant to keep things explicit.
    """

    def __init__(
        self,
        *,
        tfrecord_paths: List[str],
        feature_key: str = "value",
        test_tfrecord_paths: Optional[List[str]] = None,
        gzip: bool = False,
        time_step: int = 100,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.1,
        shuffle_buffer: Optional[int] = None,
        scale_minmax: bool = True,
    ) -> None:
        if not tfrecord_paths:
            raise ValueError("tfrecord_paths must be provided")
        self.tfrecord_paths = tfrecord_paths
        self.test_tfrecord_paths = test_tfrecord_paths
        self.feature_key = feature_key
        self.gzip = gzip

        self.time_step = time_step
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle_buffer = shuffle_buffer  # if None, will default to train size
        self.scale_minmax = scale_minmax

        self.min: float | None = None
        self.max: float | None = None

    # --------------------------- parsing ---------------------------
    def _parse_example(self) -> tf.types.experimental.ConcreteFunction:
        # Build a parse function that extracts a single scalar float from feature_key.
        key = self.feature_key

        @tf.function
        def _fn(serialized: tf.Tensor) -> tf.Tensor:
            ex = tf.io.parse_single_example(
                serialized,
                {key: tf.io.FixedLenFeature([], tf.float32)}  # adjust dtype if needed
            )
            return ex[key]  # scalar float32
        return _fn

    def _load_scalar_series_ds(self, paths: List[str]) -> tf.data.Dataset:
        compression = "GZIP" if self.gzip else None
        ds = tf.data.TFRecordDataset(paths, compression_type=compression)
        parse = self._parse_example()
        ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)  # -> scalars
        return ds

    # --------------------- scaling (train-only) --------------------
    def _fit_minmax_from_ds(self, scalar_ds: tf.data.Dataset) -> None:
        # Compute min/max in streaming fashion.
        init = (tf.constant(np.inf, tf.float32), tf.constant(-np.inf, tf.float32))

        @tf.function
        def reduce_fn(state, x):
            mn, mx = state
            return (tf.minimum(mn, x), tf.maximum(mx, x))

        mn, mx = scalar_ds.reduce(init, reduce_fn)
        self.min = float(mn.numpy())
        self.max = float(mx.numpy())

    def _scale_scalar(self, x: tf.Tensor) -> tf.Tensor:
        assert self.min is not None and self.max is not None
        denom = (self.max - self.min) + tf.constant(1e-12, tf.float32)
        return (x - self.min) / denom

    # ---------------------- windowing utilities --------------------
    def _window_xy(self, scalar_ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Turn a scalar series dataset into (X[T,1], y[]) pairs for LSTM.
        Produces a dataset of (X, y) where:
          - X shape: [time_step, 1]
          - y shape: []
        """
        T = self.time_step
        # Create sliding windows length T+1, shift 1, then batch them for materialization
        ds = scalar_ds.window(T + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(T + 1))
        # Split window into (X, y) and add feature dim
        ds = ds.map(
            lambda arr: (tf.expand_dims(arr[:-1], -1), arr[-1]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds

    def _shuffle_batch_prefetch(self, ds: tf.data.Dataset, shuffle: bool) -> tf.data.Dataset:
        if shuffle:
            # Try to pick a healthy shuffle buffer
            buf = self.shuffle_buffer if self.shuffle_buffer else 10_000
            ds = ds.shuffle(buf, reshuffle_each_iteration=True)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _split_train_val(self, ds_xy: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        # Use dataset cardinality to split train/val deterministically.
        card = tf.data.experimental.cardinality(ds_xy).numpy()
        if card < 0:  # UNKNOWN
            # Fallback: materialize count once (not ideal for massive sets, but robust).
            card = sum(1 for _ in ds_xy)
        n_val = int(max(1, round(card * self.validation_split)))
        n_tr = max(1, card - n_val)
        ds_tr = ds_xy.take(n_tr)
        ds_val = ds_xy.skip(n_tr)
        return ds_tr, ds_val

    # ------------------------- model & metrics ---------------------
    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.time_step, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(1),
        ])
        model.compile(loss="mse", optimizer="adam", metrics=[_r2_metric])
        return model

    def _inverse_np(self, data: np.ndarray) -> np.ndarray:
        if not self.scale_minmax:
            return data
        assert self.min is not None and self.max is not None
        return data * (self.max - self.min) + self.min

    @staticmethod
    def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        eps = 1e-12
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + eps)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        # nrmse will be computed after inverse (original scale) using range max-min:
        nrmse = rmse  # placeholder; caller fixes to / (max-min)
        return {"r2": float(r2), "mape": float(mape), "nrmse": float(nrmse)}

    # ------------------------------ train --------------------------
    def train(self) -> TrainingResult:
        # 1) Load train (and test) scalar series
        tr_scalars = self._load_scalar_series_ds(self.tfrecord_paths)
        te_scalars = None
        if self.test_tfrecord_paths:
            te_scalars = self._load_scalar_series_ds(self.test_tfrecord_paths)

        # 2) Fit scaling on TRAIN ONLY, then scale
        if self.scale_minmax:
            self._fit_minmax_from_ds(tr_scalars)
            tr_scalars = tr_scalars.map(lambda x: self._scale_scalar(x), num_parallel_calls=tf.data.AUTOTUNE)
            if te_scalars is not None:
                te_scalars = te_scalars.map(lambda x: self._scale_scalar(x), num_parallel_calls=tf.data.AUTOTUNE)

        # 3) Window → (X,y) datasets
        ds_tr_xy = self._window_xy(tr_scalars)
        ds_te_xy = self._window_xy(te_scalars) if te_scalars is not None else None

        # 4) Train/Val split (on TRAIN windows)
        ds_tr, ds_val = self._split_train_val(ds_tr_xy)

        # 5) Final input pipelines
        ds_tr = self._shuffle_batch_prefetch(ds_tr, shuffle=True)
        ds_val = self._shuffle_batch_prefetch(ds_val, shuffle=False)
        ds_te = self._shuffle_batch_prefetch(ds_te_xy, shuffle=False) if ds_te_xy is not None else None

        # 6) Train
        model = self._build_model()
        model.fit(ds_tr, validation_data=ds_val, epochs=self.epochs, verbose=0)

        # 7) Collect true labels and predictions (for metrics & inverse scaling)
        def collect_y(ds: tf.data.Dataset) -> np.ndarray:
            ys = []
            for _, y in ds.unbatch().as_numpy_iterator():
                ys.append(y)
            return np.asarray(ys, dtype=np.float32)

        y_train = collect_y(ds_tr)
        y_val = collect_y(ds_val)
        y_test = collect_y(ds_te) if ds_te is not None else None

        train_pred = model.predict(ds_tr, verbose=0).squeeze()
        val_pred = model.predict(ds_val, verbose=0).squeeze()
        test_pred = model.predict(ds_te, verbose=0).squeeze() if ds_te is not None else None

        # 8) Inverse-scale back to original units (if scaled)
        y_train_inv = self._inverse_np(y_train)
        y_val_inv = self._inverse_np(y_val)
        train_pred_inv = self._inverse_np(train_pred)
        val_pred_inv = self._inverse_np(val_pred)

        metrics = {
            "train": self._metrics_dict(y_train_inv, train_pred_inv),
            "val": self._metrics_dict(y_val_inv, val_pred_inv),
        }

        if y_test is not None and test_pred is not None:
            y_test_inv = self._inverse_np(y_test)
            test_pred_inv = self._inverse_np(test_pred)
            metrics["test"] = self._metrics_dict(y_test_inv, test_pred_inv)

        # fix nrmse with range (only meaningful if scaling was enabled)
        if self.scale_minmax and self.min is not None and self.max is not None:
            rng = (self.max - self.min) + 1e-12
            for split in metrics.values():
                split["nrmse"] = float(split["nrmse"] / rng)

        # 9) Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(__file__).parent / "saved" / f"mls_lstm_{timestamp}"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_dir, include_optimizer=False)

        return TrainingResult(model_dir=str(save_dir), metrics=metrics)
