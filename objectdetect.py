import cv2
import torch

# Load YOLOv5 pre-trained model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model
model.conf = 0.5  # Set confidence threshold

# Start video capture
cap = cv2.VideoCapture(0)  # Access the webcam

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB for YOLO model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(rgb_frame)

    # Parse the detection results into a pandas DataFrame
    detections = results.pandas().xyxy[0]

    # Draw bounding boxes and labels
    for _, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        label = detection['name']
        confidence = detection['confidence']

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Add label and confidence
        cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Real-Time Object Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
