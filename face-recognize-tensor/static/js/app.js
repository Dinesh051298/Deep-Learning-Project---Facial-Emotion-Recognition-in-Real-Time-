class EmotionRecognitionApp {
    constructor() {
        this.video = document.getElementById('videoElement');
        this.cameraSelect = document.getElementById('cameraSelect');
        this.startButton = document.getElementById('startButton');
        this.emotionLabel = document.getElementById('emotionLabel');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceText = document.getElementById('confidenceText');
        this.emotionDisplay = document.getElementById('emotionDisplay');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.fpsCounter = document.getElementById('fpsCounter');
        this.detectionCounter = document.getElementById('detectionCounter');
        this.accuracyIndicator = document.getElementById('accuracyIndicator');
        this.emojiGrid = document.getElementById('emojiGrid');
        this.predictionIntervalSelect = document.getElementById('predictionInterval');
        this.videoOverlay = document.getElementById('videoOverlay');

        this.emotionEmojis = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😄',
            'neutral': '😐',
            'sad': '😢',
            'surprise': '😲'
        };

        this.stats = {
            frameCount: 0,
            detectionCount: 0,
            lastFpsUpdate: Date.now(),
            fps: 0,
            predictionInterval: null,
            isCameraActive: false
        };

        this.init();
    }

    init() {
        this.initEmojiGrid();
        this.bindEvents();
        this.requestCameraPermission();
    }

    bindEvents() {
        this.startButton.addEventListener('click', () => this.toggleCamera());
        this.predictionIntervalSelect.addEventListener('change', () => this.updatePredictionInterval());
    }

    initEmojiGrid() {
        const emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
        this.emojiGrid.innerHTML = '';

        for (const emotion of emotions) {
            const emojiItem = document.createElement('div');
            emojiItem.className = 'emoji-item';
            emojiItem.id = `emoji-${emotion}`;
            emojiItem.innerHTML = `
                <div class="emoji">${this.emotionEmojis[emotion]}</div>
                <div>${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</div>
            `;
            this.emojiGrid.appendChild(emojiItem);
        }
    }

    async requestCameraPermission() {
        try {
            this.updateStatus('Requesting camera access...', 'processing');
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            for (const track of stream.getTracks()) {
                track.stop();
            }
            await this.listCameras();
            this.updateStatus('Ready to start camera', 'ready');
        } catch (err) {
            console.error("Camera permission denied:", err);
            this.updateStatus('Camera access denied', 'error');
            this.cameraSelect.innerHTML = '<option value="">Please allow camera access and refresh</option>';
        }
    }

    async listCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cameras = devices.filter(device => device.kind === 'videoinput');
            this.cameraSelect.innerHTML = '';

            if (cameras.length === 0) {
                this.cameraSelect.innerHTML = '<option value="">No cameras found</option>';
                return;
            }

            for (const cam of cameras) {
                const option = document.createElement('option');
                option.value = cam.deviceId;
                option.text = cam.label || `Camera ${this.cameraSelect.length + 1}`;
                this.cameraSelect.appendChild(option);
            }
        } catch (err) {
            console.error('Error listing cameras:', err);
            this.updateStatus('Error accessing cameras', 'error');
        }
    }

    async toggleCamera() {
        if (this.stats.isCameraActive) {
            await this.stopCamera();
        } else {
            await this.startCamera();
        }
    }

    async startCamera() {
        const deviceId = this.cameraSelect.value;
        if (!deviceId) {
            alert("Please select a camera first!");
            return;
        }

        try {
            this.startButton.disabled = true;
            this.startButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
            this.updateStatus('Initializing camera...', 'processing');

            this.video.srcObject = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: {exact: deviceId},
                    width: {ideal: 640},
                    height: {ideal: 480},
                    frameRate: {ideal: 30}
                }
            });
            this.stats.isCameraActive = true;

            this.updateStatus('Camera active - Analyzing emotions', 'active');
            this.startButton.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
            this.startButton.disabled = false;

            this.startPredictionLoop();

        } catch (err) {
            console.error('Error starting camera:', err);
            alert("Failed to start camera. Please check permissions and try again.");
            this.updateStatus('Camera error', 'error');
            this.startButton.disabled = false;
            this.startButton.innerHTML = '<i class="fas fa-play"></i> Start Camera';
            this.stats.isCameraActive = false;
        }
    }

    async stopCamera() {
        this.clearPredictionInterval();

        if (this.video.srcObject) {
            const stream = this.video.srcObject;
            const tracks = stream.getTracks();

            for (const track of tracks) {
                track.stop();
            }

            this.video.srcObject = null;
        }

        this.stats.isCameraActive = false;
        this.updateStatus('Camera stopped', 'ready');
        this.startButton.innerHTML = '<i class="fas fa-play"></i> Start Camera';

        this.resetUI();
    }

    clearPredictionInterval() {
        if (this.stats.predictionInterval) {
            clearInterval(this.stats.predictionInterval);
            this.stats.predictionInterval = null;
        }
    }

    startPredictionLoop() {
        const interval = parseInt(this.predictionIntervalSelect.value) || 500;
        this.clearPredictionInterval();

        this.stats.predictionInterval = setInterval(() => {
            if (!this.stats.isCameraActive) {
                this.clearPredictionInterval();
                return;
            }
            this.captureAndPredict();
        }, interval);
    }

    updatePredictionInterval() {
        if (this.stats.isCameraActive) {
            this.startPredictionLoop();
        }
    }

    async captureAndPredict() {
        this.updateFps();

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = this.video.videoWidth || 640;
        canvas.height = this.video.videoHeight || 480;

        try {
            context.drawImage(this.video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.updateUI(data);

        } catch (err) {
            console.error("Prediction error:", err);
            this.handlePredictionError();
        }
    }

    updateUI(data) {
        if (data.emotion === "No Face Detected") {
            this.emotionLabel.textContent = "No Face Detected";
            this.confidenceFill.style.width = '0%';
            this.confidenceText.textContent = 'Confidence: 0%';
            this.emotionDisplay.className = 'emotion-display no-face';
            this.videoOverlay.style.borderColor = 'var(--error-color)';
            this.clearEmojiHighlights();
        } else if (data.emotion && data.confidence) {
            const emotionName = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
            const confidencePercent = (data.confidence * 100).toFixed(1);

            this.emotionLabel.textContent = emotionName;
            this.confidenceFill.style.width = `${confidencePercent}%`;
            this.confidenceText.textContent = `Confidence: ${confidencePercent}%`;
            this.emotionDisplay.className = 'emotion-display detected';
            this.videoOverlay.style.borderColor = 'var(--success-color)';

            this.updateEmojiHighlight(data.emotion);
            this.updateStats(confidencePercent);
        } else if (data.error) {
            this.emotionLabel.textContent = "Prediction Error";
            this.confidenceFill.style.width = '0%';
            this.confidenceText.textContent = 'Service unavailable';
            this.emotionDisplay.className = 'emotion-display no-face';
            this.videoOverlay.style.borderColor = 'var(--error-color)';
        }
    }

    handlePredictionError() {
        this.emotionLabel.textContent = "Network Error";
        this.confidenceFill.style.width = '0%';
        this.confidenceText.textContent = 'Connection failed';
        this.emotionDisplay.className = 'emotion-display no-face';
        this.videoOverlay.style.borderColor = 'var(--error-color)';
    }

    updateEmojiHighlight(emotion) {
        this.clearEmojiHighlights();
        const activeEmoji = document.getElementById(`emoji-${emotion}`);
        if (activeEmoji) {
            activeEmoji.classList.add('active');
        }
    }

    clearEmojiHighlights() {
        for (const item of document.querySelectorAll('.emoji-item')) {
            item.classList.remove('active');
        }
    }

    updateStats(confidencePercent) {
        this.stats.detectionCount++;
        this.detectionCounter.textContent = this.stats.detectionCount.toLocaleString();
        this.accuracyIndicator.textContent = `${confidencePercent}%`;
    }

    updateFps() {
        this.stats.frameCount++;
        const now = Date.now();
        if (now - this.stats.lastFpsUpdate >= 1000) {
            this.stats.fps = this.stats.frameCount;
            this.stats.frameCount = 0;
            this.stats.lastFpsUpdate = now;
            this.fpsCounter.textContent = this.stats.fps;
        }
    }

    updateStatus(message, state = 'ready') {
        this.statusText.textContent = message;
        this.statusDot.className = 'status-dot';

        switch (state) {
            case 'active':
                this.statusDot.classList.add('active');
                break;
            case 'processing':
                this.statusDot.classList.add('processing');
                break;
            case 'error':
                this.statusDot.style.background = 'var(--error-color)';
                break;
            default:
                this.statusDot.style.background = 'var(--warning-color)';
        }
    }

    resetUI() {
        this.emotionLabel.textContent = '---';
        this.confidenceFill.style.width = '0%';
        this.confidenceText.textContent = 'Confidence: 0%';
        this.emotionDisplay.className = 'emotion-display';
        this.videoOverlay.style.borderColor = 'transparent';
        this.clearEmojiHighlights();

        this.stats.detectionCount = 0;
        this.detectionCounter.textContent = '0';
        this.accuracyIndicator.textContent = '0%';
        this.fpsCounter.textContent = '0';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EmotionRecognitionApp();
});