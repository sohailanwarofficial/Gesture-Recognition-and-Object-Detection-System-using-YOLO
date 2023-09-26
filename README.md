# Gesture Recognition and Object Detection System

This repository contains code for a Gesture Recognition and Object Detection System using YOLOv5 and a custom-trained model. The system can detect hand gestures and objects in real-time using a webcam or process images.

## Prerequisites

Before running the code, make sure you have the following requirements installed:

- Python 3.x
- Git
- Pip

## Installation

1. Clone the YOLOv5 repository:

   ```bash
   cd /content/drive/MyDrive/
   !git clone https://github.com/ultralytics/yolov5
   ```

2. Install required packages:

   ```bash
   cd /content/drive/MyDrive/yolov5
   !pip install -r requirements.txt
   !pip install roboflow
   ```

3. Import the Roboflow dataset for gesture recognition:

   ```python
   from roboflow import Roboflow

   # Initialize Roboflow API
   rf = Roboflow(api_key="YOUR_API_KEY")

   # Specify the project and dataset
   project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
   dataset = project.version(1).download("yolov5")
   ```

## Training and Detection

### Gesture Recognition Training

Train the custom YOLOv5 model for gesture recognition:

```bash
!python train.py --img 640 --batch 16 --epochs 50 --data /content/drive/MyDrive/yolov5/gesture-detection-system-1/data.yaml --weights yolov5s.pt
```

Detect gestures in an image:

```python
import torch
from PIL import Image

# Load your custom-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt')

# Load the source image
image_path = "PATH_TO_IMAGE"  # Replace with the path to your image
img = Image.open(image_path)

# Perform inference on the image
results = model(img)

# Load class labels
with open('/content/drive/MyDrive/yolov5/gesture-detection-system-1/data.yaml', 'r') as f:
    class_labels = f.read().splitlines()[1:]

# Process the results
for result in results.pandas().xyxy[0].values:
    label = int(result[5])
    class_name = class_labels[label]
    confidence = result[4]
    print(f"Detected class: {class_name}, Confidence: {confidence:.2f}")

# To visualize the image with bounding boxes
results.show()
```

### Object Detection with Webcam

Run real-time object detection using a webcam:

```python
import cv2
import torch

# Start webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
width, height = 640, 640  # Adjust to a supported webcam resolution
cap.set(3, width)
cap.set(4, height)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt')

# Object classes
classNames = ['backward', 'forward', 'left', 'right', 'stop']

while True:
    success, img = cap.read()

    if not success:
        print("Error: Failed to read a frame from the webcam.")
        break

    # Perform inference using YOLOv5
    results = model(img, size=width)

    # Process the results
    for r in results.pred[0]:
        # Bounding box coordinates
        x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])

        # Confidence
        confidence = round(r[4].item(), 2)

        # Class name
        cls = int(r[5].item())
        detected_class = classNames[cls]

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        org = (x1, y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(img, detected_class, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        # Press 'q' to quit the loop and close the webcam
        break
    elif key == ord('c'):
        # Press 'c' to close the webcam without quitting the script
        cap.release()
        cv2.destroyAllWindows()
        break

# Close the webcam
cap.release()
cv2.destroyAllWindows()
```

### Object Detection with Arduino Integration

Real-time object detection with Arduino integration:

1. Install the `pyserial` library:

   ```bash
   pip install pyserial
   ```

2. Update the serial port and baud rate in the script.

3. Run the code:

   ```python
   python object_detection_with_arduino.py
   ```

This system detects objects using YOLOv5 and sends the detected class name to an Arduino via serial communication.

---

Feel free to modify and adapt this code to suit your specific project needs. Make sure to replace placeholders (e.g., `"YOUR_API_KEY"`, `"YOUR_WORKSPACE"`, `"YOUR_PROJECT"`, `"PATH_TO_IMAGE"`) with your own values. Enjoy using the Gesture Recognition and Object Detection System!
