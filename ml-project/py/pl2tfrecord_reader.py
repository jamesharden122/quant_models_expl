# Requires: tensorflow
import tensorflow as tf
from typing import Dict, Optional, List

_DTYPE_MAP = {
    "float32": tf.float32, "float": tf.float32, "f32": tf.float32,
    "int64": tf.int64, "int": tf.int64, "i64": tf.int64,
    "string": tf.string, "bytes": tf.string, "str": tf.string,
    "bool": tf.bool,
} 

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
    return ds.repeat().take(200) 


def _mk_seq_feature() -> tf.io.FixedLenSequenceFeature:
    return tf.io.FixedLenSequenceFeature([], dtype=tf.float32)

def load_time_series_regress_tfrecord_dataset(
    paths: List[str],
    feature_spec: Dict[str, str],          # e.g. {"x":"float32","y":"int64"}
    label: Optional[str] = "label",        # context key containing the target
    label_dtype: str = "float32",          # "float32" | "int64" | "string"
    batch_size: int = 256,
    shuffle: bool = True,
    gzip: bool = False,
):
    compression = "GZIP" if gzip else None
    ds = tf.data.TFRecordDataset(paths, compression_type=compression)

    # Identify sequence vs context keys
    seq_keys   = [k for k, t in feature_spec.items() if t == "float32"]
    ctx_int64  = [k for k, t in feature_spec.items() if t == "int64"]
    ctx_bytes  = [k for k, t in feature_spec.items() if t == "string"]

    # Specs for parse_single_sequence_example
    def _mk_seq_feature() -> tf.io.FixedLenSequenceFeature:
        return tf.io.FixedLenSequenceFeature([], dtype=tf.float32)

    seq_spec = {k: _mk_seq_feature() for k in seq_keys}
    ctx_spec: Dict[str, tf.io.FixedLenFeature] = {}

    # Label spec
    if label is not None:
        if label_dtype == "float32":
            ctx_spec[label] = tf.io.FixedLenFeature([], tf.float32)
        elif label_dtype == "int64":
            ctx_spec[label] = tf.io.FixedLenFeature([], tf.int64)
        elif label_dtype == "string":
            ctx_spec[label] = tf.io.FixedLenFeature([], tf.string)
        else:
            raise ValueError(f"Unsupported label_dtype: {label_dtype!r}")

    # Other context keys (don’t overwrite label)
    for k in ctx_int64:
        if k != label:
            ctx_spec[k] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
    for k in ctx_bytes:
        if k != label:
            ctx_spec[k] = tf.io.FixedLenFeature([], tf.string, default_value=b"")

    # Common optional fields (don’t overwrite label)
    if "t_end" not in ctx_spec:
        ctx_spec["t_end"] = tf.io.FixedLenFeature([], tf.int64, default_value=0)
    if "group" not in ctx_spec:
        ctx_spec["group"] = tf.io.FixedLenFeature([], tf.string, default_value=b"")

    # Parse → (x_seq[T,F], y)
    def _parse(ex):
        context, seq_feats = tf.io.parse_single_sequence_example(
            ex, context_features=ctx_spec, sequence_features=seq_spec
        )
        x_seq = tf.stack([seq_feats[k] for k in seq_keys], axis=-1) if seq_keys else tf.zeros([0, 0], tf.float32)
        if label is None:
            return x_seq
        y = context[label]
        return (x_seq, y)

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)

    # ---- Infer T (and F) from the first element, then lock shapes ----
    # Eager-only peek (works in notebooks / normal TF2 eager mode)
    x0, y0 = next(iter(ds.take(1)))
    T = x0.shape[0] or int(tf.shape(x0)[0].numpy())
    F = x0.shape[1] or int(tf.shape(x0)[1].numpy())

    # Force shape for all elements so element_spec becomes (None, T, F)
    def _lock_shape(x, y):
        x = tf.ensure_shape(x, (T, F))
        # Cast y to float32 if you want Keras-friendly targets regardless of stored dtype:
        # y = tf.cast(y, tf.float32)
        return x, y

    ds = ds.map(_lock_shape, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch with fixed batch size; drop last partial to keep static shapes
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


# --- pl2tfrecord_reader.py ---

def load_sharpe_timeseries_dataset(
    paths: List[str],
    feature_spec: Dict[str, str],   # name -> dtype string (in desired, insertion order)
    time_steps: Optional[int] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    gzip: bool = False,
    include_cost: bool = False,
) -> tf.data.Dataset:
    print("hello we got here")
    """
    Returns (X, y_true) where:
      X:      (B, T, F) float32, features stacked in insertion-order of feature_spec
      y_true: (B, 3) or (B, 4): [r_next, sigma_t, boundary_mask (, cost)]

    Expects SequenceExample with:
      - FeatureLists: names from feature_spec (scalar per timestep)
      - Context: label (r_next), sigma_t, inst_id (bytes), optional cost
    """
    # --- normalize / validate feature spec
    feat_names = list(feature_spec.keys())                # stable feature order
    try:
        feat_tf_dtypes = [_DTYPE_MAP[feature_spec[n].lower()] for n in feat_names]
    except KeyError as e:
        raise ValueError(f"Unsupported dtype in feature_spec: {e}")
    F = len(feat_names)

    # --- specs for parse_single_sequence_example
    # sequence features: scalar per timestep, stored in their native dtype
    seq_spec = {
        name: tf.io.FixedLenSequenceFeature([], dtype=dtype)
        for name, dtype in zip(feat_names, feat_tf_dtypes)
    }
    print("hello we got here 2")
    # context features
    ctx_spec = {
        "label":   tf.io.FixedLenFeature([], tf.float32),  # r_next
        "sigma": tf.io.FixedLenFeature([], tf.float32),
        "inst_id": tf.io.FixedLenFeature([], tf.string),
    }
    if include_cost:
        ctx_spec["cost"] = tf.io.FixedLenFeature([], tf.float32)
    print("hello we got here 3")
    compression = "GZIP" if gzip else None
    ds = tf.data.TFRecordDataset(paths, compression_type=compression)
    print("hello we got here 4")
    # --- parse → (x_seq[T,F], r_next, sigma_t, inst_id(, cost))
    def _parse(ex):
        context, seq_feats = tf.io.parse_single_sequence_example(
            ex, context_features=ctx_spec, sequence_features=seq_spec
        )
        # cast each feature to float32, then stack to (T, F) in declared order
        seq_list = [tf.cast(seq_feats[name], tf.float32) for name in feat_names]
        x_seq = tf.stack(seq_list, axis=-1)  # (T, F)

        r_next  = context["label"]
        sigma_t = context["sigma"]
        inst_id = context["inst_id"]
        if include_cost:
            return (x_seq, r_next, sigma_t, inst_id, context["cost"])
        else:
            return (x_seq, r_next, sigma_t, inst_id)

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)

    # --- lock/ensure time dimension (T)
    if time_steps is not None:
        T = int(time_steps)
    else:
        sample = next(iter(ds.take(1)), None)
        if sample is None:
            raise ValueError("Empty TFRecord dataset; cannot infer time_steps.")
        x0 = sample[0]  # (T, F)
        T = x0.shape[0] if x0.shape[0] is not None else int(tf.shape(x0)[0].numpy())

    def _lock_T(*elts):
        x = elts[0]
        x = tf.ensure_shape(x, (T, F))
        return (x, *elts[1:])

    ds = ds.map(_lock_T, num_parallel_calls=tf.data.AUTOTUNE)

    # --- batch before computing boundary mask (mask is along batch axis)
    ds = ds.batch(batch_size, drop_remainder=False)

    # --- boundary mask + pack labels to y_true
    def _pack(*elts):
        """
        Inputs (batched):
          x: (B, T, F)
          r: (B,)
          s: (B,)
          inst: (B,) bytes
          cost (optional): (B,)
        Output:
          (x, y_true) where y_true = [r, s, bmask(, cost)]
        """
        x, r, s, inst = elts[:4]

        # Hash bytes → int for robust comparisons
        inst_i64 = tf.strings.to_hash_bucket_fast(inst, num_buckets=2**31 - 1)
        inst_prev = tf.roll(inst_i64, shift=1, axis=0)

        # boundary at batch start or instrument change
        bmask = tf.where(tf.not_equal(inst_i64, inst_prev), 1.0, 0.0)
        bmask = tf.tensor_scatter_nd_update(bmask, indices=[[0]], updates=[1.0])

        if include_cost:
            cost_vec = tf.cast(elts[4], tf.float32)
            y_true = tf.stack(
                [tf.cast(r, tf.float32), tf.cast(s, tf.float32), bmask, cost_vec],
                axis=-1
            )
        else:
            y_true = tf.stack(
                [tf.cast(r, tf.float32), tf.cast(s, tf.float32), bmask],
                axis=-1
            )
        return x, y_true

    ds = ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

