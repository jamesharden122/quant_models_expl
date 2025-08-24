import json
from datetime import datetime
from pathlib import Path
import importlib

import polars as pl
import tensorflow as tf

from utils import df_to_tfds


def train_from_polars(df: pl.DataFrame, cfg_json: str) -> str:
    """Train a TensorFlow model from a Polars DataFrame.

    Args:
        df: Input features and target column.
        cfg_json: JSON string with training configuration.

    Returns:
        Path to the SavedModel directory.
    """
    cfg = json.loads(cfg_json)
    target = cfg["target"]
    epochs = cfg.get("epochs", 1)
    batch = cfg.get("batch", 256)
    outdir = Path(cfg.get("outdir", "models"))
    outdir.mkdir(parents=True, exist_ok=True)

    ds = df_to_tfds(df, target=target, batch=batch)
    input_dim = df.drop(target).width
    model_module = cfg["model_module"]
    model_fn = cfg.get("model_fn", "build_model")
    module = importlib.import_module(model_module)
    build_model = getattr(module, model_fn)
    model = build_model(input_dim, cfg)
    model.fit(ds, epochs=epochs)

    model_dir = outdir / datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(model_dir)
    return str(model_dir)
