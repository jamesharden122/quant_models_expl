from __future__ import annotations
# mls_lstm_trainer.py  — minimal trainer with internal split + per-epoch saving
from sharpeloss import SharpeLossExCostLongOnlyRoll
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import sys
import subprocess
import textwrap

# If you already have these in base_trainer, keep imports as-is.
from base_trainer import BaseTrainer, TrainingResult


def _r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


class MLSLSTMTrainer(BaseTrainer):
    """
    Minimal LSTM trainer:
      - Accepts a pre-windowed dataset: (X, y) where X: (B,T,F), y: (B,) or (B,1).
      - Internally splits into train/val(/test) without altering the data.
      - Saves a checkpoint of the FULL model (.keras) after EVERY epoch.
    """

    def __init__(
        self,
        full_ds: tf.data.Dataset,
        time_steps: Optional[int] = None,      # if None, infer from element_spec
        input_dim: Optional[int] = None,       # if None, infer from element_spec
        val_split: float = 0.1,
        test_split: float = 0.1,               # set > 0 to carve test from full_ds
        test_ds: Optional[tf.data.Dataset] = None,  # or pass an explicit test set
        epochs: int = 10,
        batch_size: Optional[int] = None,      # leave None if ds already batched
        verbose: int = 0,
        shuffle_before_split: bool = True,
        seed: Optional[int] = 42,
        # --- Saving options ---
        save_every_epoch: bool = True,         # saves model after each epoch
        save_weights_only: bool = False,       # False = full model (.keras); True = weights only
        monitor: str = "val_loss",             # used if you later set save_best_only=True
        save_best_only: bool = False,          # False keeps all epochs; True keeps only best
        run_name: Optional[str] = None,        # optional folder suffix for grouping
        sharpe_loss_params: Dict = dict( 
            sigma_tgt=0.15, ann_factor=252.0,
            cost=0.0002, vol=True,
            lambda_exposure=0.0, lambda_l2=0.0,
            )
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # hide GPUs for this process
        assert 0.0 <= val_split < 1.0, "val_split must be in [0,1)"
        assert 0.0 <= test_split < 1.0, "test_split must be in [0,1)"
        assert val_split + test_split < 1.0, "val_split + test_split must be < 1.0"

        self.full_ds = full_ds
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.test_ds = test_ds
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle_before_split = shuffle_before_split
        self.seed = seed

        self.save_every_epoch = save_every_epoch
        self.save_weights_only = save_weights_only
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.sharpe_loss_params = sharpe_loss_params
        # Prepare a unique run directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"mls_lstm_{ts}" if not run_name else f"{run_name}"#_{ts}"
        self.save_dir = (Path(__file__).parent / "saved" / base)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Infer (T, F) from dataset spec if not provided
        if time_steps is None or input_dim is None:
            x_spec, _ = self._get_spec(full_ds)
            if len(x_spec.shape) == 3:      # (B,T,F)
                inferred_T = x_spec.shape[1]
                inferred_F = x_spec.shape[2]
            else:                            # (T,F)
                inferred_T = x_spec.shape[0]
                inferred_F = x_spec.shape[1]
            if time_steps is None:
                if inferred_T is None:
                    raise ValueError("time_steps could not be inferred; please specify it.")
                time_steps = int(inferred_T)
            if input_dim is None:
                if inferred_F is None:
                    raise ValueError("input_dim could not be inferred; please specify it.")
                input_dim = int(inferred_F)

        self.time_steps = int(time_steps)
        self.input_dim  = int(input_dim)
        self.model = self._build_model(self.time_steps, self.input_dim)

    @staticmethod
    def _get_spec(ds: tf.data.Dataset):
        es = ds.element_spec
        if isinstance(es, tuple) and len(es) == 2:
            return es[0], es[1]
        raise ValueError("Dataset must yield tuples (X, y).")

    def _build_model(self, time_steps: int, input_dim: int) -> Sequential:
        model = Sequential([
            Input(shape=(time_steps, input_dim)),
                LSTM(64, return_sequences=True,  unroll=True),
                LSTM(64, return_sequences=True,  unroll=True),
                LSTM(64, return_sequences=False, unroll=True),
                Dense(1, activation="sigmoid"),  # long-only positions X_t ∈ [0,1]
            ])
        model.compile(
        optimizer=tf.keras.optimizers.Adam(),
            loss=SharpeLossExCostLongOnlyRoll(
                sigma_tgt=self.sharpe_loss_params["sigma_tgt"],        # match units to your sigma_t labels
                ann_factor=self.sharpe_loss_params["ann_factor"],      # change if not daily
                cost=self.sharpe_loss_params["cost"],           # example: 2 bps per unit traded weight
                vol_scale_turnover=self.sharpe_loss_params["vol"],
                lambda_exposure=self.sharpe_loss_params["lambda_exposure"],
                lambda_l2=self.sharpe_loss_params["lambda_l2"],
            ),
            run_eagerly=False,
        )
        return model
 
    def _maybe_batch(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds if self.batch_size is None else ds.batch(self.batch_size)

    @staticmethod
    def _dataset_len(ds: tf.data.Dataset) -> int:
        card = tf.data.experimental.cardinality(ds).numpy()
        if card < 0:  # UNKNOWN
            card = sum(1 for _ in ds)
        return int(card)

    def _split(self, ds: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
        n = self._dataset_len(ds)
        if self.shuffle_before_split:
            ds = ds.shuffle(buffer_size=min(10000, max(1, n)), seed=self.seed, reshuffle_each_iteration=False)

        n_test = int(round(n * self.test_split))
        n_val  = int(round(n * self.val_split))
        n_train = max(1, n - n_val - n_test)

        ds_train = ds.take(n_train)
        remainder = ds.skip(n_train)
        ds_val = remainder.take(n_val)
        ds_test_auto = remainder.skip(n_val).take(n_test) if n_test > 0 else None

        ds_test_final = self.test_ds if self.test_ds is not None else ds_test_auto
        return ds_train, ds_val, ds_test_final

    @staticmethod
    def _collect_y(ds: tf.data.Dataset) -> np.ndarray:
        ys = []
        for _, y in ds.unbatch().as_numpy_iterator():
            ys.append(y)
        return np.asarray(ys, dtype=np.float32).squeeze()

    @staticmethod
    def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        eps = 1e-12
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + eps)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)
        return {"r2": float(r2), "mape": mape, "nrmse": rmse}
        
    def _eval_trading_metrics(
        self,
        ds: tf.data.Dataset,
        ann_factor: float = 252.0,
    ) -> Dict[str, float]:
        """
        Compute realized trading metrics from dataset context and model predictions.
        Assumes y columns: [r_next, sigma_t, boundary_mask, (optional) cost].
        """
        if ds is None:
            return None

        rets = []
        costs = []
        masks = []
        preds = []

        # Collect in consistent order
        for X, y in ds:
            y_np = y.numpy()
            r_next = y_np[:, 0]                         # (B,)
            boundary = y_np[:, 2] if y_np.shape[1] > 2 else np.ones_like(r_next)
            # Per-step cost: from y if present else use loss default
            step_cost = y_np[:, 3] if y_np.shape[1] > 3 else np.full_like(r_next, self.sharpe_loss_params["cost"])

            p = self.model.predict_on_batch(X).squeeze()   # (B,)
            preds.append(p)
            rets.append(r_next)
            masks.append(boundary)
            costs.append(step_cost)

        p = np.concatenate(preds).astype(np.float32)            # positions in [0,1]
        r = np.concatenate(rets).astype(np.float32)
        m = np.concatenate(masks).astype(np.float32)
        c = np.concatenate(costs).astype(np.float32)

        # Long-only turnover (absolute change in position)
        # prepend first pos to align length
        turnover = np.abs(np.diff(p, prepend=p[:1]))

        # Masked realized returns (e.g., 0 where boundary_mask==0)
        ret_gross = m * (p * r)
        ret_net   = ret_gross - c * turnover

        # Stats (avoid tiny denom)
        eps = 1e-12
        mu  = float(np.mean(ret_net))
        sd  = float(np.std(ret_net, ddof=1) + eps)
        sharpe = float(np.sqrt(ann_factor) * (mu / sd))
        avg_pos = float(np.mean(p))
        avg_turn = float(np.mean(turnover))

        return {
            "mu_daily": mu,
            "sigma_daily": sd,
            "sharpe_ann": sharpe,
            "avg_position": avg_pos,
            "avg_turnover": avg_turn,
        }
    def _build_callbacks(self):
        cbs = []

        # Always keep a CSV log (epoch, loss, val_loss, etc.)
        csv_cb = tf.keras.callbacks.CSVLogger(str(self.save_dir / "training_log.csv"), append=False)
        cbs.append(csv_cb)

        if self.save_every_epoch:
            # Save FULL model (.keras) or weights after EACH epoch
            suffix = "weights.h5" if self.save_weights_only else "keras"
            ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.save_dir / f"epoch_{{epoch:03d}}.{suffix}"),
                save_freq="epoch",
                save_weights_only=self.save_weights_only,
                save_best_only=self.save_best_only,  # keep all unless True
                monitor=self.monitor,
                mode="auto",
            )
            cbs.append(ckpt_cb)

        return cbs

    def train(self) -> TrainingResult:
        # Split → train/val(/test)
        ds_train, ds_val, ds_test = self._split(self.full_ds)

        # Respect user batching; prefetch for throughput
        ds_train = self._maybe_batch(ds_train).prefetch(tf.data.AUTOTUNE)
        ds_val   = self._maybe_batch(ds_val).prefetch(tf.data.AUTOTUNE)
        ds_test  = self._maybe_batch(ds_test).prefetch(tf.data.AUTOTUNE) if ds_test is not None else None

        # Fit with callbacks (per-epoch saving happens here)
        callbacks = self._build_callbacks()
        self.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=callbacks,
        )

        # Metrics (no inverse-scaling here; dataset is assumed final)
        def eval_ds(ds):
            return self._eval_trading_metrics(ds, ann_factor=self.sharpe_loss_params["ann_factor"]) 

        metrics = {
            "train": eval_ds(ds_train),
            "val":   eval_ds(ds_val),
        }
        if ds_test is not None:
            metrics["test"] = eval_ds(ds_test)

        # Optional final export (sometimes handy)
        
        # Optional final export (sometimes handy)
        final_path = self.save_dir / "final_model.keras"
        self.model.save(final_path, include_optimizer=False)

        # ---- CPU-only ONNX export in a fresh subprocess (avoids GPU/driver checks) ----
        
        code = textwrap.dedent(f"""
                import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                import numpy as np

                # --- NumPy>=1.24 / 2.0 shims (remove once tf2onnx is upgraded) ---
                if not hasattr(np, "object"): np.object = object

                import tensorflow as tf
                import tf2onnx, pathlib

                model = tf.keras.models.load_model(r'{final_path}', compile=False)
                spec = [tf.TensorSpec([None, {self.time_steps}, {self.input_dim}], tf.float32, name='inputs')]
                fn = tf.function(model)
                tf2onnx.convert.from_function(
                    fn,
                    input_signature=spec,
                    opset=13,
                    output_path=str(pathlib.Path(r'{self.save_dir}')/'final_model.onnx'),
                )
                """)


        subprocess.run([sys.executable, "-c", code], check=True)

        return {"model_dir": str(self.save_dir), "metrics": metrics}
