import tensorflow as tf


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
