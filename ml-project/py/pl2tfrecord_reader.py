# Requires: tensorflow
import tensorflow as tf
from typing import Dict, Optional, List


def _mk_fixedlen(dtype: str) -> tf.io.FixedLenFeature:
    if dtype == "float32":
        return tf.io.FixedLenFeature([], tf.float32)
    if dtype == "int64":
        return tf.io.FixedLenFeature([], tf.int64)
    if dtype == "string":
        return tf.io.FixedLenFeature([], tf.string)
    raise ValueError(f"Unsupported dtype in feature_spec: {dtype}")


def load_tfrecord_dataset(
    paths: List[str],
    feature_spec: Dict[str, str],
    label: Optional[str] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    gzip: bool = False,
):
    """
    Return a parsed tf.data.Dataset from TFRecord files.
    feature_spec: {"col": "float32"|"int64"|"string", ...}
    label: optional label column (will be popped out of features).
    """
    compression = "GZIP" if gzip else None
    ds = tf.data.TFRecordDataset(paths, compression_type=compression)

    parse_spec = {k: _mk_fixedlen(v) for k, v in feature_spec.items()}

    def _parse(ex):
        parsed = tf.io.parse_single_example(ex, parse_spec)
        y = None
        if label is not None and label in parsed:
            y = parsed.pop(label)
        # Optional int64 -> int32 cast (GPU friendlier)
        for k, v in parsed.items():
            if v.dtype == tf.int64:
                parsed[k] = tf.cast(v, tf.int32)
        if y is not None and y.dtype == tf.int64:
            y = tf.cast(y, tf.int32)
        return (parsed, y) if y is not None else parsed

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
