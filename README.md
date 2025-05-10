# Hand Gesture Recognition System (HGRS)

A real-time hand gesture recognition system capable of identifying numerical hand gestures (0-9) using computer vision and deep learning techniques.

## Project Overview

This Hand Gesture Recognition System (HGRS) is designed to recognize numerical hand gestures (0-9) in real-time using a webcam. The system uses convolutional neural networks (CNNs) for classification and offers multiple interfaces for interaction, including a basic region-of-interest (ROI) approach, an advanced hand detection approach using MediaPipe, and a graphical user interface for easier interaction.

### Key Features

- Real-time recognition of numerical hand gestures (0-9)
- Multiple interface options (basic ROI, MediaPipe-based, GUI)
- High accuracy classification using custom CNN architecture
- Data augmentation to improve model robustness
- Comprehensive training and evaluation metrics
- Supports unknown/neutral gesture class

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- MediaPipe (optional, for advanced hand detection)
- Tkinter and PIL (optional, for GUI)

## Dataset

The system is trained on a dataset of hand gesture images for numbers 0-9 plus an "unknown" class. The dataset should be organized as follows:

```
Sign Language for Numbers/
├── 0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
...
├── 9/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── unknown/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

To use your own dataset, update the `DATA_PATH` variable in the code to point to your dataset directory.

## Project Structure

The project is structured into four main milestones:

1. **Data Collection, Preprocessing, and Exploration**: Loading and preparing the dataset
2. **Model Development and Training**: Building and training the CNN model
3. **Real-Time Gesture Recognition and Deployment**: Implementing the real-time recognition system
4. **MLOps Implementation and Model Monitoring**: (In progress)

## Usage

### 1. Data Preprocessing

- Load the dataset
- Explore the class distribution
- Visualize sample images
- Preview data augmentation techniques
- Preprocess and split the data into training, validation, and test sets
- Save the preprocessed data to the 'data/' directory

### 2. Model Training

- Load the preprocessed data
- Build a custom CNN model
- Train the model with data augmentation
- Evaluate the model on the test set
- Generate visualizations for performance analysis
- Save the trained model to 'models/hand_gesture_model.keras'

### 3. Real-Time Recognition

You'll be prompted to select a mode:
1. Basic Recognition (with ROI)
2. Advanced Recognition (with hand detection, if MediaPipe is available)
3. GUI Application (if Tkinter and PIL are available)

#### Basic Mode Instructions:
- Place your hand within the green box
- Show a number gesture (0-9)
- Press 'q' to quit
- Press 'r' to reset the ROI position

#### Advanced Mode Instructions:
- Show your hand to the camera
- The system will automatically detect and track your hand
- Press 'q' to quit

#### GUI Mode Instructions:
- Select the desired mode from the dropdown
- Click "Start" to begin recognition
- Show your hand gesture to the camera
- Click "Stop" to end recognition

## Model Architecture

The project uses a custom CNN architecture with the following features:
- Multiple convolutional blocks with batch normalization
- Max pooling and dropout for regularization
- L2 regularization on convolutional and dense layers
- Final softmax layer for multi-class classification

## Performance

On the test set, the model achieves:
- Accuracy: ~92%
- F1-Score: ~0.93 (averaged across classes)

## Troubleshooting

### Common Issues

1. **Webcam not working**:
   - Ensure your webcam is properly connected
   - Check if other applications are using the webcam
   - Try changing the camera index in the code (e.g., `cv2.VideoCapture(1)`)

2. **MediaPipe not available**:
   - Install MediaPipe with: `pip install mediapipe`

3. **GUI not available**:
   - Ensure Tkinter is installed (usually comes with Python)
   - Install PIL with: `pip install Pillow`

4. **Model file not found**:
   - Make sure you've run the training script first
   - Check if the model file exists at 'models/hand_gesture_model.keras'

## Future Improvements

- Implement MLOps for model monitoring and versioning
- Add support for more gesture types
- Improve hand detection in challenging lighting conditions
- Develop mobile application deployment

## Acknowledgments

- TensorFlow and MediaPipe teams for their excellent libraries
- Contributors to the various open-source packages used in this project