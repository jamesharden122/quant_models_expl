import polars as pl
import tensorflow as tf


def df_to_tfds(df: pl.DataFrame, target: str, batch: int = 256, shuffle: bool = True) -> tf.data.Dataset:
    """Convert a Polars DataFrame into a tf.data.Dataset."""
    y = df[target].to_numpy()
    x = df.drop(target).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(df))
    return ds.batch(batch)


def build_model(input_dim: int, cfg: dict) -> tf.keras.Model:
    """Build a simple dense model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(cfg.get("hidden", 64), activation="relu"),
        tf.keras.layers.Dense(1, activation=cfg.get("output_activation", "linear")),
    ])
    model.compile(
        optimizer=cfg.get("optimizer", "adam"),
        loss=cfg.get("loss", "mse"),
        metrics=cfg.get("metrics", ["mse"]),
    )
    return model
