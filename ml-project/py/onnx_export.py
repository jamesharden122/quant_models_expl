import tf2onnx
import tensorflow as tf


def savedmodel_to_onnx_bytes(saved_model_dir: str) -> bytes:
    """Convert a TensorFlow SavedModel to ONNX and return serialized bytes."""
    model_proto, _ = tf2onnx.convert.from_saved_model(saved_model_dir, opset=17)
    return model_proto.SerializeToString()


def keras_to_onnx_bytes(model_path: str) -> bytes:
    """Convert a Keras `.keras` model file to ONNX bytes."""
    model = tf.keras.models.load_model(model_path)
    model_proto, _ = tf2onnx.convert.from_keras(model, opset=17)
    return model_proto.SerializeToString()
