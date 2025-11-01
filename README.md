# Real-Time Facial Emotion Recognition using Flask and PyTorch

This repository contains a web-based real-time facial emotion recognition system built with **Flask**, **PyTorch**, and **OpenCV**.
The application detects faces from webcam or uploaded images, classifies the detected emotions using a deep learning model, and serves predictions through a web interface or API.

---

## Overview

The project integrates a **deep learning emotion classifier** trained using **PyTorch** with a **Flask web application** for real-time inference.
It can recognize the following seven emotion classes:

```
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

The model processes grayscale facial images and predicts the emotion based on features learned from datasets such as FER2013.

---

## Project Structure

```
face-recognize-pytorch/
│
├── app.py                     # Flask application for inference
├── torch_model.py             # Model architecture (EmotionCNN)
├── best_emotion_model.pth     # Trained model weights
├── class_labels.json          # Emotion class label mappings
├── requirements.txt           # Dependencies
│
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
│
└── templates/
    └── index.html             # Web interface for real-time emotion recognition
```

---

## Features

* Real-time face detection using OpenCV’s Haar Cascade classifier
* Emotion classification using a trained CNN model in PyTorch
* Web-based inference via Flask with REST API support
* JSON response for API integration or browser-based predictions
* Logging and health-check endpoints for debugging and monitoring

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Dinesh051298/realtime-facial-emotion-detection.git
cd "realtime-facial-emotion-detection/face-recognize-pytorch"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dependencies

| Package                  | Purpose                                  |
| ------------------------ | ---------------------------------------- |
| torch, torchvision       | Model architecture and inference         |
| flask                    | Web server and API routes                |
| opencv-python            | Face detection and image preprocessing   |
| numpy                    | Array manipulation                       |
| Pillow                   | Image handling for transformations       |
| matplotlib, scikit-learn | Model evaluation and plotting (optional) |

Manual installation example:

```bash
pip install torch torchvision flask opencv-python numpy Pillow
```

---

## Model Description

The model used in this project is a **Convolutional Neural Network (CNN)** implemented in PyTorch.
It contains four convolutional blocks with **Batch Normalization**, **ReLU activations**, **Dropout**, and **MaxPooling** layers, followed by fully connected layers for classification.

**Training details:**

* Input size: 48x48 grayscale images
* Optimizer: Adam
* Loss function: CrossEntropyLoss
* Dataset: FER2013 (or any dataset with emotion-labeled images)
* Achieved accuracy: approximately 67% on test data

The model and class labels are saved as:

```
best_emotion_model.pth
class_labels.json
```

---

## Running the Application

1. Ensure the following files are present in the same directory:

   * `best_emotion_model.pth`
   * `class_labels.json`
   * `torch_model.py`

2. Start the Flask application:

   ```bash
   python app.py
   ```

3. Open the application in your web browser:

   ```
   http://127.0.0.1:5000/
   ```

The web interface will display live webcam feed or allow image uploads for real-time emotion recognition.

---

## API Endpoints

| Endpoint   | Method | Description                                                     |
| ---------- | ------ | --------------------------------------------------------------- |
| `/`        | GET    | Renders the main web page                                       |
| `/predict` | POST   | Accepts an image (base64 format) and returns emotion prediction |
| `/health`  | GET    | Returns server and model status information                     |

### Example Request to `/predict`

**Request (JSON):**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
}
```

**Response (JSON):**

```json
{
  "emotion": "happy",
  "confidence": 0.945,
  "face_detected": true,
  "face_coordinates": {
    "x": 120,
    "y": 80,
    "width": 150,
    "height": 150
  }
}
```

---

## Logging

The application uses Python’s `logging` module for runtime monitoring.

Example console output:

```
2025-11-01 14:32:10,142 - INFO - Loading PyTorch model and class labels...
2025-11-01 14:32:11,812 - INFO - Model loaded successfully
2025-11-01 14:32:12,123 - INFO - Face detector initialized
2025-11-01 14:32:12,456 - INFO - Server running on http://localhost:5000
```

Logging helps track model loading, prediction steps, and any runtime errors.

---

## Health Check Example

To verify if the server and model are properly loaded, run:

```bash
curl http://127.0.0.1:5000/health
```

**Sample Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "face_detector_loaded": true,
  "device": "cuda"
}
```

---

## Face Detection Details

* The application uses OpenCV’s **Haar Cascade Classifier** for frontal face detection.
* Automatically selects the largest detected face for prediction.
* Works on both color and grayscale images.
* Falls back to predefined cascade file paths if the default is unavailable.

---

## Prediction Pipeline

1. Receive base64 image data via API or frontend
2. Decode and convert the image into an OpenCV-compatible format
3. Detect the face region using Haar Cascade
4. Preprocess the cropped face (resize, normalize, convert to tensor)
5. Perform inference using the trained PyTorch model
6. Return the predicted emotion and confidence score

---

## Troubleshooting

| Issue                  | Possible Solution                                                                 |
| ---------------------- | --------------------------------------------------------------------------------- |
| Model not found        | Ensure `best_emotion_model.pth` is in the same directory as `app.py`              |
| Haar cascade not found | Reinstall OpenCV or specify a valid path to `haarcascade_frontalface_default.xml` |
| CUDA error             | Use CPU by setting `device = torch.device('cpu')`                                 |
| No face detected       | Ensure sufficient lighting and clear frontal view of the face                     |
| Failed to decode image | Verify that the base64 string is valid and correctly formatted                    |
