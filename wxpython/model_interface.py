"""model_interface.py: Interface for YOLO models"""

import torch
import cv2
import numpy as np
from model_config import (
    MODEL1_PATH,
    MODEL2_PATH,
    MODEL1_CONFIDENCE_THRESHOLD,
    MODEL1_IOU_THRESHOLD,
    MODEL2_CONFIDENCE_THRESHOLD,
    MODEL2_IOU_THRESHOLD,
    DEBUG_MODE
)

class ModelInterface:
    """Base class for YOLO model interface"""
    
    def __init__(self, model_path, confidence_threshold, iou_threshold):
        """Initialize YOLO model"""
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if DEBUG_MODE:
            print(f"Loaded YOLO model from {model_path}")
            print(f"Confidence threshold: {confidence_threshold}")
            print(f"IOU threshold: {iou_threshold}")

    def predict(self, image):
        """Run prediction on input image"""
        results = self.model(image)
        return results

class Module1(ModelInterface):
    """Interface for module1: Detects part position and crops minimum unit"""
    
    def __init__(self):
        super().__init__(MODEL1_PATH, MODEL1_CONFIDENCE_THRESHOLD, MODEL1_IOU_THRESHOLD)
        
    def detect_and_crop(self, image):
        """Detect part position and crop minimum unit"""
        results = self.predict(image)
        
        # Get bounding boxes
        boxes = results.xyxy[0].cpu().numpy()
        
        # Filter by confidence
        boxes = boxes[boxes[:, 4] >= self.confidence_threshold]
        
        if DEBUG_MODE:
            print(f"Detected {len(boxes)} parts in the image")
        
        # Process each detected part
        cropped_images = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)
            
        return boxes, cropped_images

class Module2(ModelInterface):
    """Interface for module2: Detects defects in minimum unit"""
    
    def __init__(self):
        super().__init__(MODEL2_PATH, MODEL2_CONFIDENCE_THRESHOLD, MODEL2_IOU_THRESHOLD)
        
    def detect_defects(self, image):
        """Detect defects in minimum unit image"""
        results = self.predict(image)
        
        # Get predictions
        predictions = results.xyxy[0].cpu().numpy()
        
        # Filter by confidence
        predictions = predictions[predictions[:, 4] >= self.confidence_threshold]
        
        if DEBUG_MODE:
            print(f"Detected {len(predictions)} defects in the image")
            
        return predictions
