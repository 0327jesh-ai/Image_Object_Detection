# ==========================================================
# IMPORTING REQUIRED LIBRARIES
# ==========================================================

# OpenCV library:
# - Used for video reading, drawing bounding boxes
# - Provides DNN module to run YOLOv4
import cv2

# NumPy library:
# - Used for numerical operations
# - Helps in handling arrays and confidence scores
import numpy as np


# ==========================================================
# DEFINING FILE PATHS
# ==========================================================

# Path to input video containing pedestrians
video_path = "Pedestrian.mp4"

# Path to pretrained YOLOv4 weights file
weights_path = "./yolov4/yolov4.weights"

# Path to YOLOv4 configuration file (network architecture)
config_path = "./yolov4/yolov4.cfg"

# Path to COCO class labels file
names_path = "./yolov4/coco.names"


# ==========================================================
# LOADING CLASS NAMES
# ==========================================================

# Open coco.names file and read all object class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Note:
# COCO dataset has 80 classes
# Class ID 0 corresponds to "person"


# ==========================================================
# LOADING YOLOv4 MODEL
# ==========================================================

# Load YOLOv4 network using OpenCV DNN module
net = cv2.dnn.readNet(weights_path, config_path)

# Get all layer names from the YOLO network
layer_names = net.getLayerNames()

# Identify output layers (YOLO detection layers)
output_layers = [
    layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()
]


# ==========================================================
# OPENING THE INPUT VIDEO
# ==========================================================

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check whether the video is successfully opened
print("Video opened:", cap.isOpened())


# ==========================================================
# SETUP VIDEO WRITER TO SAVE OUTPUT
# ==========================================================

# Define codec for MP4 video format
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create VideoWriter object to save output video
out = cv2.VideoWriter(
    "output_people.mp4",
    fourcc,
    20,
    (int(cap.get(3)), int(cap.get(4)))
)


# ==========================================================
# PROCESS VIDEO FRAME BY FRAME
# ==========================================================

while True:

    # Read a single frame from the video
    ret, frame = cap.read()

    # If no frame is returned, video has ended
    if not ret:
        break

    # Get height and width of the frame
    height, width, _ = frame.shape


    # ======================================================
    # PREPROCESS FRAME FOR YOLO (BLOB CREATION)
    # ======================================================

    # Convert frame into blob
    # - Scale pixel values to range [0,1]
    # - Resize image to 416x416
    # - Convert BGR to RGB
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    # Set the blob as input to the YOLO network
    net.setInput(blob)

    # Perform forward pass to get detections
    outputs = net.forward(output_layers)


    # ======================================================
    # INITIALIZE LISTS FOR DETECTIONS
    # ======================================================

    # List to store bounding boxes
    boxes = []

    # List to store confidence scores
    confidences = []


    # ======================================================
    # PROCESS YOLO DETECTIONS
    # ======================================================

    for output in outputs:
        for detection in output:

            # Extract class confidence scores
            scores = detection[5:]

            # Get the class ID with highest confidence
            class_id = np.argmax(scores)

            # Get confidence value of detected class
            confidence = scores[class_id]


            # ==================================================
            # FILTER ONLY PERSON CLASS
            # ==================================================

            # Class ID 0 corresponds to "person"
            # Confidence threshold set to 0.5
            if class_id == 0 and confidence > 0.5:

                # Convert center coordinates to absolute values
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                # Convert width and height to absolute values
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Store bounding box and confidence
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))


    # ======================================================
    # APPLY NON-MAXIMUM SUPPRESSION
    # ======================================================

    # Remove overlapping boxes and keep best ones
    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        0.5,
        0.4
    )


    # ======================================================
    # DRAW BOUNDING BOXES AND LABELS
    # ======================================================

    for i in indexes.flatten():

        # Get bounding box coordinates
        x, y, w, h = boxes[i]

        # Draw rectangle around detected person
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display label and confidence score
        cv2.putText(
            frame,
            f"Person {confidences[i]:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ======================================================
    # SAVE THE PROCESSED FRAME
    # ======================================================

    out.write(frame)


# ==========================================================
# RELEASE ALL RESOURCES
# ==========================================================

# Release input video
cap.release()

# Release output video writer
out.release()

# Destroy any OpenCV windows (safe cleanup)
cv2.destroyAllWindows()
