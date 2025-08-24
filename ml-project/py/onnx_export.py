import tf2onnx


def savedmodel_to_onnx_bytes(saved_model_dir: str) -> bytes:
    """Convert a TensorFlow SavedModel to ONNX and return serialized bytes."""
    model_proto, _ = tf2onnx.convert.from_saved_model(saved_model_dir, opset=17)
    return model_proto.SerializeToString()
