# src/tools/classification_tools.py
import os
import joblib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from langchain.tools import tool


# Load pre-trained models (adjust paths if needed)
CLASSIFIER_PATH = "../models/classifier.joblib"
KMEANS_PATH = "../models/kmeans.joblib"

if not os.path.exists(KMEANS_PATH):
    raise FileNotFoundError(f"kmeans model not found: {KMEANS_PATH}")
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError(f"Classifier model not found: {CLASSIFIER_PATH}")

kmeans = joblib.load(open(KMEANS_PATH, "rb"))
classifier = joblib.load(open(CLASSIFIER_PATH, "rb"))

# SIFT detector
sift = cv2.SIFT_create()


# Your feature extraction functions (adapted from your notebook)
def color_Moments(img):
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    colorFeatures = np.array(
        [np.mean(R), np.std(R), np.mean(G), np.std(G), np.mean(B), np.std(B)]
    )
    colorFeatures /= np.mean(colorFeatures) + 1e-8
    return colorFeatures


def hsvHistogramFeatures(image):
    imageHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 2, 2], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten()
    hist /= np.sum(hist) + 1e-8
    return hist.reshape(-1)


def textureFeatures(img):
    im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(
        im, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    v = np.array(
        [
            graycoprops(glcm, "contrast")[0][0],
            graycoprops(glcm, "correlation")[0][0],
            graycoprops(glcm, "energy")[0][0],
            graycoprops(glcm, "homogeneity")[0][0],
        ]
    )
    v /= np.sum(v) + 1e-8
    return v


def shapeFeatures(img):
    im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    shapeFeat = cv2.HuMoments(cv2.moments(im)).flatten()
    shapeFeat /= np.mean(shapeFeat) + 1e-8
    return shapeFeat


def extractSIFTDescriptors(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(im_gray, None)
    return des if des is not None else np.array([])


def computeBoVWHistogram(descriptors, kmeans):
    hist = np.zeros(kmeans.n_clusters)
    if descriptors.size > 0:
        labels = kmeans.predict(descriptors)
        for l in labels:
            hist[l] += 1
    hist /= np.sum(hist) + 1e-8
    return hist


def getFeatures(img):
    features = color_Moments(img)
    features = np.concatenate((features, hsvHistogramFeatures(img)), axis=None)
    features = np.concatenate((features, textureFeatures(img)), axis=None)
    features = np.concatenate((features, shapeFeatures(img)), axis=None)
    des = extractSIFTDescriptors(img)
    feats_sift = computeBoVWHistogram(des, kmeans)
    feats_sift /= np.linalg.norm(feats_sift) + 1e-8
    features = np.concatenate((features, feats_sift), axis=None)
    features /= np.linalg.norm(features) + 1e-8
    return features


@tool("Classify Image with ML Model")
def classify_image(image_path: str) -> str:
    """
    Loads an image, extracts features (color, HSV, texture, shape, SIFT+BoVW),
    and predicts 'Anomaly' or 'Normal' using the pre-trained .pkl model.
    """
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"

    # Load image (BGR) and convert to RGB
    img = cv2.imread(image_path)
    if img is None:
        return f"Error: Failed to load image at {image_path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract features
    features = getFeatures(img)

    # Predict
    pred = classifier.predict(features.reshape(1, -1))[0]
    label = "Anomaly" if pred == 1 else "Normal"  # Adjust based on your model's labels

    # Optional: Confidence if model supports it
    confidence = None
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(features.reshape(1, -1))[0]
        confidence = max(proba) * 100  # Max probability as %

    result = f"Prediction: {label}"
    if confidence:
        result += f" (Confidence: {confidence:.1f}%)"

    # Save features/prediction to outputs/ (optional)
    os.makedirs("outputs", exist_ok=True)
    np.save(f"outputs/features_{os.path.basename(image_path)}.npy", features)
    with open(f"outputs/prediction_{os.path.basename(image_path)}.txt", "w") as f:
        f.write(result)

    return (
        result
        + f"\nFeatures saved to outputs/features_{os.path.basename(image_path)}.npy"
    )