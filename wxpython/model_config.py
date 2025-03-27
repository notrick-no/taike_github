"""model_config.py: Global configuration for YOLO models"""

# Path to YOLO model weights
# Modify these paths as needed
MODEL1_PATH = "./YOLODataset/yolo11n.pt"  # model1 weights file
MODEL2_PATH = "./YOLODataset/yolov8n-seg.pt"  # model2 weights file

# Model-specific parameters
MODEL1_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for model1
MODEL1_IOU_THRESHOLD = 0.45  # IOU threshold for model1
MODEL2_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for model2
MODEL2_IOU_THRESHOLD = 0.45  # IOU threshold for model2

# Other settings
FRAME_REFRESH_INTERVAL = 30  # Interval in ms to refresh frames in video
DEBUG_MODE = True  # Set True to enable debug print statements
