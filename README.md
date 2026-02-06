# YOLOv4 Object Detection (People & Objects in Video)

This project implements **real-time object detection** using **YOLOv4 pretrained weights** and **OpenCV**.  
It detects **people (pedestrians)** and other common objects in a video using the **COCO dataset**.

The project is designed to work smoothly in **VS Code**, **Linux/WSL**, or **Windows**, and is beginner-friendly with clear structure.

---

## Features

- Uses **YOLOv4 pretrained weights**
- Detects **people and 80+ objects** from the COCO dataset
- Works on **video files** (`Pedestrian.mp4`)
- Bounding boxes with **labels and confidence scores**
- Optimized for real-time detection
- Simple and clean Python implementation

---

## Objects Detected (COCO Dataset)

YOLOv4 with COCO can detect:
- Person (Pedestrian)
- Car, Bus, Truck, Bicycle, Motorcycle
- Handbag, Backpack, Umbrella
- Traffic lights, Stop signs
- Animals (dog, cat, horse, etc.)
- Many everyday objects found in public places

Perfect for **crowded areas**, **shopping malls**, **bus stops**, and **footbridges**.

---

## Project Structure
Image_Object_Detection/
│
├── Pedestrian.mp4 # Input video
├── detect_people.py # Main detection script
├── coco.names # COCO class labels
├── yolov4/
│ ├── yolov4.cfg # YOLOv4 configuration
│ └── yolov4.weights # Pretrained weights
└── venv/ # Python virtual environment

### Pedestrian.mp4
- Input video file
- Contains pedestrians and crowded public scenes
- Used for object detection testing

### detect_people.py
- Main Python script of the project
- Loads YOLOv4 model and COCO classes
- Reads video frame by frame
- Detects people and other objects
- Draws bounding boxes and labels
- Displays the output video

### coco.names
- Text file containing **80 object class names**
- Based on COCO dataset
- Includes: person, car, bus, handbag, backpack, etc.
- YOLO uses this file to assign labels to detected objects

### yolov4/
This folder stores YOLOv4 model files.

#### yolov4.cfg
- YOLOv4 network configuration file
- Defines neural network architecture
- Required to build the detection model

#### yolov4.weights
- Pretrained YOLOv4 weights
- Trained on COCO dataset
- Used to detect objects without training from scratch

###  venv/
- Python virtual environment
- Keeps project dependencies isolated
- Avoids package conflicts with system Python

---

## Execution Flow (Simple)

1. Load YOLOv4 config and weights
2. Load class labels from coco.names
3. Open video file (Pedestrian.mp4)
4. Read video frame by frame
5. Run object detection on each frame
6. Filter detections by confidence
7. Draw bounding boxes and labels
8. Display output video in real time

-----------------------------------------------------------------------
Author
Jeshwanth
