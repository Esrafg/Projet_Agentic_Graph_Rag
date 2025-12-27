import tensorflow as tf
import numpy as np
from PIL import Image

def load_and_preprocess_image(filepath: str, target_size=(224, 224)):
    """Load and preprocess a single image for the model (same as training)"""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return img.numpy()  # Returns numpy array in [0,1]