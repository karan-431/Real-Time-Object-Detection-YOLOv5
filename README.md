# Real-Time Object Detection using YOLOv5

This project implements real-time object detection using the YOLOv5 model in Python. It captures live video feed from a webcam and detects objects using a pre-trained YOLOv5 model.

## 🚀 Features
- Uses **YOLOv5s** pre-trained model from Ultralytics.
- Real-time object detection via **OpenCV** and **PyTorch**.
- Displays detected objects with bounding boxes and confidence scores.
- Works with live webcam feed.

## 🛠️ Installation
### Prerequisites
Ensure you have Python (>=3.7) installed and set up.

### Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install opencv-python pandas
pip install ultralytics
```

## 🔧 Usage
Run the following command to start real-time object detection:
```bash
python detect.py
```

## 📜 Code Explanation
- **Loads YOLOv5s model** from PyTorch Hub.
- **Captures video** from the webcam.
- **Performs inference** on each frame.
- **Draws bounding boxes** around detected objects with labels.
- **Displays real-time feed** with detections.
- **Exits on 'q' key press**.

## 📌 Example Output
When the script runs, you will see real-time object detection with bounding boxes and labels drawn around detected objects.

## 🏗️ Future Improvements
- Support for custom YOLOv5 models.
- Integration with video file input.
- Faster inference using GPU acceleration.

