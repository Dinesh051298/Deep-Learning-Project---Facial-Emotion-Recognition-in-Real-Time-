import base64
import json
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms

app = Flask(__name__)

# ---------------------- Setup Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------- Load Model & Labels ----------------------
logger.info("Loading PyTorch model and class labels...")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

try:
    # Load the saved checkpoint
    checkpoint = torch.load("best_emotion_model.pth", map_location=device)

    # First, we need to know the number of classes to initialize the model
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
    num_classes = len(class_labels)

    # Import or define the model architecture (you'll need to have EmotionCNN class available)
    from torch_model import EmotionCNN  # Adjust import path as needed

    # Initialize model
    model = EmotionCNN(num_classes=num_classes)

    # Handle DataParallel if it was used during training
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        # Model was saved with DataParallel, create a new state dict without 'module.' prefix
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()  # Set to evaluation mode

    logger.info("PyTorch model loaded successfully")

except Exception as e:
    logger.error(f"Error loading PyTorch model: {e}")
    raise

try:
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
    logger.info(f"Class labels loaded: {class_labels}")
except Exception as e:
    logger.error(f"Error loading class labels: {e}")
    raise

# ---------------------- Load Face Detector ----------------------
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    logger.info(f"Loading cascade from: {cascade_path}")

    if not os.path.exists(cascade_path):
        logger.error(f"Cascade file not found at: {cascade_path}")
        possible_paths = [
            cascade_path,
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                cascade_path = path
                logger.info(f"Found cascade at: {path}")
                break
        else:
            raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise Exception("Failed to load cascade classifier")

    logger.info("Face cascade classifier loaded successfully")

except Exception as e:
    logger.error(f"Error loading face detector: {e}")
    raise

# ---------------------- Preprocessing Transform ----------------------
# Define the same transform used during training
preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def detect_face(frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect faces in the frame and return the first face ROI
    """
    try:
        frame_np = np.array(frame)

        if len(frame_np.shape) == 3:
            if frame_np.shape[2] == 3:
                gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            elif frame_np.shape[2] == 4:
                gray = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2GRAY)
            else:
                gray = frame_np[:, :, 0]
        else:
            gray = frame_np

        gray = np.ascontiguousarray(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return None, None

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        face_roi = gray[y:y + h, x:x + w]

        return face_roi, (x, y, w, h)

    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return None, None


def preprocess_face(face_roi: np.ndarray) -> torch.Tensor:
    """
    Preprocess face ROI for emotion prediction (PyTorch version)
    """
    try:
        # Apply the same transform as during training
        processed = preprocess_transform(face_roi)
        # Add batch dimension
        processed = processed.unsqueeze(0)
        # Move to device
        processed = processed.to(device)

        return processed

    except Exception as e:
        logger.error(f"Error in face preprocessing: {e}")
        raise


def predict_emotion(processed_face: torch.Tensor) -> Tuple[str, float]:
    """
    Predict emotion using PyTorch model
    """
    try:
        with torch.no_grad():
            outputs = model(processed_face)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            emotion_idx = predicted.item()
            confidence_score = confidence.item()

            if isinstance(class_labels, list):
                emotion = class_labels[emotion_idx]
            elif isinstance(class_labels, dict):
                emotion = class_labels.get(str(emotion_idx), class_labels.get(emotion_idx, "Unknown"))
            else:
                emotion = "Unknown"

            return emotion, confidence_score

    except Exception as e:
        logger.error(f"Error in emotion prediction: {e}")
        return "Unknown", 0.0


# ---------------------- Routes ----------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "face_detector_loaded": not face_cascade.empty(),
        "device": str(device)
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        if 'image' not in data:
            return jsonify({"error": "No image data in request"}), 400

        _, img_data = data['image'].split(',', 1) if ',' in data['image'] else ('', data['image'])
        img_bytes = base64.b64decode(img_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.warning("Failed to decode image")
            return jsonify({
                "emotion": "No Face Detected",
                "confidence": 0.0,
                "face_detected": False
            })

        face_roi, face_coords = detect_face(frame)

        if face_roi is None:
            logger.info("No face detected in image")
            return jsonify({
                "emotion": "No Face Detected",
                "confidence": 0.0,
                "face_detected": False
            })

        processed_face = preprocess_face(face_roi)
        emotion, confidence = predict_emotion(processed_face)

        logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.4f})")

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "face_detected": True,
            "face_coordinates": {
                "x": int(face_coords[0]),
                "y": int(face_coords[1]),
                "width": int(face_coords[2]),
                "height": int(face_coords[3])
            } if face_coords else None
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "emotion": "Error",
            "confidence": 0.0,
            "face_detected": False
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    logger.info("Starting Flask server...")
    logger.info("Server is running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)