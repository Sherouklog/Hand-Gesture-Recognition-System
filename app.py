import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model

# Constants
IMG_SIZE = 64
NUM_CLASSES = 11

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

class HandGestureRecognizer:
    def __init__(self, model_path, class_names=None):
        """Initialize the hand gesture recognizer with a trained model"""
        # Load model
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Set class names
        if class_names is None:
            # Default class names for numbers 0-9 and unknown
            self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown']
        else:
            self.class_names = class_names
            
        # Preprocessing parameters
        self.img_size = IMG_SIZE
        
    def preprocess_frame(self, frame):
        """Preprocess a video frame for prediction"""
        # Resize frame to expected input size
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        
        # Convert to RGB if it's BGR
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Expand dimensions to match model input shape
        expanded = np.expand_dims(normalized, axis=0)
        return expanded
        
    def predict(self, frame):
        """Make a prediction on a single frame"""
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # Make prediction
        prediction = self.model.predict(processed_frame, verbose=0)[0]
        
        # Get class index and confidence
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]
        
        # Get class name
        class_name = self.class_names[class_idx]
        
        return class_name, confidence, prediction

# Initialize the recognizer
model_path = 'models/hand_gesture_model.keras'
recognizer = None

# Check if model exists, otherwise we'll initialize it when first needed
if os.path.exists(model_path):
    recognizer = HandGestureRecognizer(model_path)
    print("Model loaded successfully!")
else:
    print(f"Warning: Model not found at {model_path}. Will attempt to load when needed.")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process image from webcam and return prediction"""
    # Check if model is loaded
    global recognizer
    
    if recognizer is None:
        if os.path.exists(model_path):
            try:
                recognizer = HandGestureRecognizer(model_path)
            except Exception as e:
                return jsonify({'error': f"Failed to load model: {str(e)}"})
        else:
            return jsonify({'error': f"Model not found at {model_path}"})
    
    try:
        # Get image data from request
        image_data = request.json.get('image', '')
        image_data = image_data.split(',')[1]  # Remove data URL header
        
        # Decode base64 image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Convert to numpy array
        frame = np.array(img)
        
        # Extract ROI (center of the image)
        h, w = frame.shape[:2]
        roi_size = min(h, w) - 20  # Smaller than the full image
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2
        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
        
        # Get prediction
        class_name, confidence, _ = recognizer.predict(roi)
        
        # Return prediction result
        return jsonify({
            'gesture': class_name,
            'confidence': float(confidence),
            'roi': {
                'x': roi_x,
                'y': roi_y,
                'size': roi_size
            }
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Check for model
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Please ensure model is available before making predictions.")
    
    # Create required directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Run Flask app
    print("Starting web server. Navigate to http://127.0.0.1:5000/ in your web browser.")
    app.run(debug=True)