from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    # Tiny synthetic regression: y = x0 * 0.3 + x1 * -0.2 + x2 * 0.1 + noise
    rng = np.random.default_rng(42)
    n = 512
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = (0.3 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2] + rng.normal(scale=0.01, size=n)).astype(
        np.float32
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=1, batch_size=64, verbose=0)

    # Save under ml-project/models/saved/<timestamped>
    repo_root = Path(__file__).resolve().parents[2]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_dir = repo_root / "ml-project" / "models" / "saved" / f"mls_lstm_{ts}"
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_dir.as_posix())
    print(save_dir.as_posix())


if __name__ == "__main__":
    main()

