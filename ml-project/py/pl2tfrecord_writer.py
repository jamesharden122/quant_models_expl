# Requires: tensorflow>=2.9, polars, numpy
import numpy as np
import polars as pl
import tensorflow as tf
from typing import Sequence, Optional, Iterable,List


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
     
    




def write_timeseries_tfrecord_from_polars(
    df: pl.DataFrame,
    path: str,
    *,
    feature_cols: List[str],
    target_col: str,
    time_col: Optional[str] = None,
    sequence_length: int = 64,
    horizon: int = 1,
    stride: int = 1,
    group_col: Optional[str] = None,
    compress: bool = False,
) -> int:
    """
    Create sliding windows and write each as a tf.train.SequenceExample.

    - feature_lists: one FeatureList per feature, length == sequence_length
    - context: scalar 'label' (float), optional 'group' (string), optional 't_end' (int64)

    Returns the number of examples written.
    """
    # Optional: keep deterministic order
    if time_col and time_col in df.columns:
        df = df.sort(time_col)

    # Restrict to needed columns up front
    needed = list(
        dict.fromkeys(
            [*feature_cols, target_col]
            + ([time_col] if time_col else [])
            + ([group_col] if group_col else [])
        )
    )
    df = df.select([c for c in needed if c in df.columns])

    # Ensure numeric dtypes we can handle; cast to float32 for features/target
    def _to_float32_frame(frame: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
        for c in cols:
            if c not in frame.columns:
                raise ValueError(f"Column '{c}' not in DataFrame.")
            dt = frame[c].dtype
            if not (dt.is_numeric() or dt == pl.Boolean):
                raise TypeError(f"Feature/target column '{c}' must be numeric/bool; got {dt}.")
        return frame.with_columns([pl.col(c).cast(pl.Float32) for c in cols])

    df = _to_float32_frame(df, feature_cols + [target_col])

    # Group iterator
    if group_col is not None and group_col in df.columns:
        # Returns a dict {group_value: DataFrame} when as_dict=True
        parts = df.partition_by(group_col, maintain_order=True, as_dict=True)
        groups = parts.items()  # Iterable[Tuple[object, pl.DataFrame]]
    else:
        groups = [(None, df)]



    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None
    n_written = 0
    with tf.io.TFRecordWriter(path, options=options) as w:
        for gid, gdf in groups:
            # Numpy arrays
            X_all = gdf.select(feature_cols).to_numpy()                         # [N, F] float32
            y_all = gdf.select(target_col).to_numpy().reshape(-1).astype(np.float32)  # [N]
            t_all = None
            if time_col and time_col in gdf.columns:
                tc = gdf.select(time_col).to_numpy().reshape(-1)
                try:
                    t_all = tc.astype("int64")  # e.g., epoch ns/µs/s
                except Exception:
                    t_all = np.arange(tc.shape[0], dtype=np.int64)

            N, F = X_all.shape
            min_len = sequence_length + horizon
            if N < min_len:
                continue

            max_start = N - min_len
            for i in range(0, max_start + 1, stride):
                x_win = X_all[i : i + sequence_length]  # [L, F]
                y_idx = i + sequence_length - 1 + horizon
                y_val = float(y_all[y_idx])

                # Build SequenceExample
                seq = tf.train.SequenceExample()

                # --- Build FeatureLists IN PLACE to avoid "Message objects may not be assigned" ---
                # For each feature column, append one Feature per timestep.
                for f_idx, fname in enumerate(feature_cols):
                    flist = seq.feature_lists.feature_list[fname]  # creates/gets entry
                    # Append L Features (each is a single float value at that timestep)
                    for t in range(sequence_length):
                        flist.feature.add().float_list.value.append(float(x_win[t, f_idx]))

                # Context fields
                seq.context.feature["label"].float_list.value.append(y_val)
                if gid is not None:
                    seq.context.feature["group"].bytes_list.value.append(str(gid).encode("utf-8"))
                if t_all is not None:
                    t_end = int(t_all[i + sequence_length - 1])
                    seq.context.feature["t_end"].int64_list.value.append(t_end)

                w.write(seq.SerializeToString())
                n_written += 1

    return n_written


# --- pl2tfrecord_writer.py ---

def write_timeseries_tfrecord_for_sharpe_from_polars(
    df: pl.DataFrame,
    path: str,
    *,
    feature_cols: List[str],
    return_col: str,              # r_{t,t+1}
    sigma_col: str,               # sigma_t (ex-ante vol at decision time)
    time_col: Optional[str] = None,
    sequence_length: int = 64,
    horizon: int = 1,
    stride: int = 1,
    cost_col: Optional[str] = None,  # optional per-sample cost
    group_col: Optional[str] = None,
    compress: bool = False,
) -> int:
    """
    Writes SequenceExample per window:
      - FeatureLists: one per feature in feature_cols (length == sequence_length)
      - Context: label=r_next, sigma_t, inst_id, (optional) cost, plus optional t_end
    """
    # 1) Order & select columns
    if time_col and time_col in df.columns:
        df = df.sort(time_col)

    needed = list(dict.fromkeys(
        [*feature_cols, return_col, sigma_col]
        + ([time_col] if time_col else [])
        + ([cost_col] if cost_col else [])
    ))
    df = df.select([c for c in needed if c in df.columns])

    # 2) Cast numerics
    def _to_float32_frame(frame: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
        for c in cols:
            if c not in frame.columns:
                raise ValueError(f"Column '{c}' not in DataFrame.")
            dt = frame[c].dtype
            if not (dt.is_numeric() or dt == pl.Boolean):
                raise TypeError(f"Column '{c}' must be numeric/bool; got {dt}.")
        return frame.with_columns([pl.col(c).cast(pl.Float32) for c in cols])

   # Build the numeric columns list (dedup!)
    num_cols = feature_cols + [return_col, sigma_col] + ([cost_col] if cost_col else [])
    print("num_cols: ",num_cols)
    # remove None and duplicates while preserving order
    num_cols = [c for i, c in enumerate(num_cols) if c and c not in num_cols[:i]]
    print("num_cols deduplicated: ",num_cols)
    df = _to_float32_frame(df, num_cols)
 
       # Group iterator
    if group_col is not None and group_col in df.columns:
        # Returns a dict {group_value: DataFrame} when as_dict=True
        parts = df.partition_by(group_col, maintain_order=True, as_dict=True)
        groups = parts.items()  # Iterable[Tuple[object, pl.DataFrame]]
    else:
        groups = [(None, df)] 

    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None
    n_written = 0
    with tf.io.TFRecordWriter(path, options=options) as w:
        for gid, gdf in groups:
            X_all = gdf.select(feature_cols).to_numpy()                            # [N,F] float32
            r_all = gdf.select(return_col).to_numpy().reshape(-1).astype(np.float32)
            s_all = gdf.select(sigma_col).to_numpy().reshape(-1).astype(np.float32)
            c_all = (gdf.select(cost_col).to_numpy().reshape(-1).astype(np.float32)
                     if cost_col and cost_col in gdf.columns else None)

            t_all = None
            if time_col and time_col in gdf.columns:
                tc = gdf.select(time_col).to_numpy().reshape(-1)
                try:
                    t_all = tc.astype("int64")
                except Exception:
                    t_all = np.arange(tc.shape[0], dtype=np.int64)

            N, F = X_all.shape
            min_len = sequence_length + horizon
            if N < min_len:
                continue
            max_start = N - min_len

            for i in range(0, max_start + 1, stride):
                x_win = X_all[i : i + sequence_length]      # [L,F]
                # Align r_next and sigma_t to decision at the END of window
                y_idx = i + sequence_length - 1 + horizon
                r_val = float(r_all[y_idx])
                s_val = float(s_all[i + sequence_length - 1])  # sigma at decision time t
                cost_val = float(c_all[y_idx]) if c_all is not None else None

                seq = tf.train.SequenceExample()

                # FeatureLists (each timestep one float)
                for f_idx, fname in enumerate(feature_cols):
                    fl = seq.feature_lists.feature_list[fname]
                    for t in range(sequence_length):
                        fl.feature.add().float_list.value.append(float(x_win[t, f_idx]))

                # Context: r_next, sigma_t, inst_id, optional cost & t_end
                seq.context.feature["label"].float_list.value.append(r_val)
                seq.context.feature["sigma"].float_list.value.append(s_val)
                # inst_id: keep stable bytes; reader can hash/cast to int later
                seq.context.feature["inst_id"].bytes_list.value.append(str(gid).encode("utf-8"))
                if cost_val is not None:
                    seq.context.feature["cost"].float_list.value.append(cost_val)
                if t_all is not None:
                    t_end = int(t_all[i + sequence_length - 1])
                    seq.context.feature["t_end"].int64_list.value.append(t_end)

                w.write(seq.SerializeToString())
                n_written += 1

    return n_written

