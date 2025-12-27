# explainability_tools.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
import os
from .model_tools import ModelLoader
from utils.image_utils import load_and_preprocess_image

# IMPORT THIS — THIS IS THE KEY FIX
# from crewai_tools import tool  # or:
from langchain.tools import tool


@tool("Explain Image with Grad-CAM")
def explain_with_gradcam(image_path: str, alpha: float = 0.5) -> dict:
    """
    Generate Grad-CAM heatmap overlay for the predicted class.
    """
    print(f"[EXPLAINABILITY TOOL ACTIVATED] Processing: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = ModelLoader.get_model()
    img_array = load_and_preprocess_image(image_path)
    input_tensor = img_array[np.newaxis, ...]

    # Get last conv layer
    last_conv_layer_name = ModelLoader.get_last_conv_layer_name(model)
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        pred_score = (
            predictions[0].numpy()
            if len(predictions.shape) == 1
            else predictions[0][0].numpy()
        )
        predicted_class = "COVID" if pred_score > 0.5 else "Normal"
        class_channel = (
            predictions if len(predictions.shape) == 1 else predictions[:, 0]
        )

    # Get gradients and pool them
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight activation maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to input size
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Load original image
    original_img = np.array(Image.open(image_path).resize((224, 224)))

    # Superimpose
    superimposed = (heatmap_colored * alpha + original_img * (1 - alpha)).astype(
        "uint8"
    )

    # Save visualization
    os.makedirs("outputs", exist_ok=True)
    gradcam_path = f"outputs/gradcam_{os.path.basename(image_path)}"
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title(f"Grad-CAM Heatmap\nPred: {predicted_class} ({pred_score:.1%})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(gradcam_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "method": "Grad-CAM",
        "predicted_class": predicted_class,
        "confidence": float(pred_score),
        "explanation_image": gradcam_path,
        "description": "Grad-CAM highlights regions in the lung X-ray that most strongly activated the model's prediction.",
        "original_image": image_path,
    }


# MAIN TOOL USED BY AGENT — NOW ONLY GRAD-CAM
@tool("Explain Image")
def explain_image(image_path: str) -> dict:
    """
    Entry point for the Explainability Agent — runs only Grad-CAM
    """
    result = explain_with_gradcam(image_path, alpha=0.5)

    return {
        "image_path": image_path,
        "prediction": f"{result['predicted_class']} ({result['confidence']:.1%} confidence)",
        "explanation": result,
        "summary": f"The model predicts **{result['predicted_class']}** with {result['confidence']:.1%} confidence. "
        f"The Grad-CAM visualization shows which areas of the chest X-ray contributed most to this decision.",
    }
