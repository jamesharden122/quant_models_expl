# Requires: tensorflow, polars
import tensorflow as tf
import polars as pl
from typing import Optional


def _bytes_feature(v):
    if isinstance(v, (list, tuple)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(v)))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _float_feature(v):
    if isinstance(v, (list, tuple)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in v]))
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))


def _int64_feature(v):
    if isinstance(v, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in v]))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def write_tfrecord_from_polars(py_df: pl.DataFrame, path: str, label: Optional[str] = None, compress: bool = False) -> None:
    """
    Write one TF Example per row of a Polars DataFrame.

    Anything non-(int/float/bool/None) is encoded as utf-8 bytes via str().
    For large data, prefer sharding and call multiple times with part paths.
    """
    if not isinstance(py_df, pl.DataFrame):
        raise TypeError("py_df must be a polars.DataFrame")

    cols = py_df.columns
    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None

    with tf.io.TFRecordWriter(path, options=options) as w:
        for row in py_df.iter_rows(named=True):
            features = {}
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    features[col] = _float_feature(val)
                elif isinstance(val, (int, bool)):
                    features[col] = _int64_feature(int(val))
                elif val is None:
                    features[col] = _bytes_feature(b"")
                else:
                    features[col] = _bytes_feature(str(val).encode("utf-8"))

            example = tf.train.Example(features=tf.train.Features(feature=features))
            w.write(example.SerializeToString())
