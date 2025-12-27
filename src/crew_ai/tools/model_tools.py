import os
import tensorflow as tf
from typing import Optional

class ModelLoader:
    _model: Optional[tf.keras.Model] = None

    @classmethod
    def get_model(cls, model_path: str = "../models/deep.keras") -> tf.keras.Model:
        if cls._model is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            print(f"Loading model from {model_path}...")
            cls._model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        return cls._model

    @classmethod
    def get_last_conv_layer_name(cls, model: tf.keras.Model) -> str:
        """Auto-detect last conv layer (works for VGG, ResNet, etc.)"""
        for layer in reversed(model.layers):
            if len(layer.output.shape) == 4 and 'conv' in layer.name.lower():
                return layer.name
        raise ValueError("No convolutional layer found for Grad-CAM")