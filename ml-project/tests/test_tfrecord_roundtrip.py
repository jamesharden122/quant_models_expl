import os
import sys


# Ensure we can import from ml-project/py
THIS_DIR = os.path.dirname(__file__)
PY_DIR = os.path.join(THIS_DIR, "..", "py")
sys.path.insert(0, os.path.abspath(PY_DIR))

import polars as pl
import tensorflow as tf

import pl2tfrecord_writer as writer
import pl2tfrecord_reader as reader


def test_tfrecord_roundtrip(tmp_path):
    df = pl.DataFrame({"x": [0.5, 1.5, 2.5], "y": [1, 0, 1]})
    out_path = tmp_path / "data.tfrecord"

    writer.write_tfrecord_from_polars(df, str(out_path), label="y", compress=False)
    assert out_path.exists() and out_path.stat().st_size > 0

    ds = reader.load_tfrecord_dataset(
        [str(out_path)], {"x": "float32", "y": "int64"}, label="y", batch_size=2, shuffle=False, gzip=False
    )

    # Take one batch and perform lightweight assertions
    for batch in ds.take(1):
        features, labels = batch
        x = features["x"].numpy()
        y = labels.numpy()
        assert x.ndim == 1 and y.ndim == 1
        assert x.shape[0] == y.shape[0] and x.shape[0] > 0
        break

