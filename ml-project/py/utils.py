import polars as pl
import tensorflow as tf


def df_to_tfds(
    df: pl.DataFrame, target: str, batch: int = 256, shuffle: bool = True
) -> tf.data.Dataset:
    """Convert a Polars DataFrame into a tf.data.Dataset."""
    y = df[target].to_numpy()
    x = df.drop(target).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(df))
    return ds.batch(batch)
