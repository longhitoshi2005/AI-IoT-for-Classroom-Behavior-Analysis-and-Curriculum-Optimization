#!/usr/bin/env python3
"""
YOLOv8 Classroom Behavior Detection - iPhone Camera Version
Connects to iPhone camera via EpocCam, Camo, or iVCam
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iphone_detection_log.txt'),
        logging.StreamHandler()
    ]
)

def connect_iphone_camera(connection_method="epoccam", ip_address=None):
    """
    Connect to iPhone camera via different methods
    
    Args:
        connection_method (str): "epoccam", "camo", "ivcam", or "direct_usb"
        ip_address (str): IP address for network connection (if needed)
    
    Returns:
        cv2.VideoCapture: Camera capture object
    """
    logging.info(f"Attempting to connect to iPhone via {connection_method}")
    
    if connection_method == "direct_usb":
        # Try USB connection first (works with EpocCam USB mode)
        logging.info("Trying USB connection...")
        cap = cv2.VideoCapture(1)  # Usually index 1 for external camera
        if cap.isOpened():
            logging.info("iPhone connected via USB")
            return cap
        
        # Try other indices
        for i in range(2, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                logging.info(f"iPhone connected via USB (index {i})")
                return cap
            cap.release()
    
    elif connection_method == "epoccam" and ip_address:
        # EpocCam WiFi connection
        url = f"http://{ip_address}:8080/video"
        logging.info(f"Connecting to EpocCam at: {url}")
        
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                logging.info("iPhone EpocCam connected via WiFi")
                return cap
        except Exception as e:
            logging.error(f"EpocCam connection failed: {e}")
    
    elif connection_method == "ivcam" and ip_address:
        # iVCam connection
        url = f"http://{ip_address}:8080/video"
        logging.info(f"Connecting to iVCam at: {url}")
        
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                logging.info("iPhone iVCam connected via WiFi")
                return cap
        except Exception as e:
            logging.error(f"iVCam connection failed: {e}")
    
    elif connection_method == "camo":
        # Camo usually appears as a USB camera
        logging.info("Looking for Camo camera...")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Test if it's actually capturing
                ret, frame = cap.read()
                if ret and frame is not None:
                    logging.info(f"iPhone Camo connected (index {i})")
                    return cap
            cap.release()
    
    logging.error("Failed to connect to iPhone camera")
    return None

def setup_iphone_connection():
    """Interactive setup for iPhone camera connection"""
    
    print(" iPhone Camera Setup Options:")
    print("1. EpocCam (USB) - Connect iPhone via cable [RECOMMENDED]")
    print("2. EpocCam (WiFi) - Both devices on same WiFi")
    print("3. Camo - Professional quality (USB)")
    print("4. iVCam - Free option (WiFi)")
    print()
    
    choice = input("Choose connection method (1-4): ").strip()
    
    if choice == "1":
        print("\n EpocCam USB Setup:")
        print("1. Install 'EpocCam Webcamera' from App Store")
        print("2. Install EpocCam drivers on PC from kinoni.com")
        print("3. Connect iPhone to PC via USB cable")
        print("4. Open EpocCam app on iPhone")
        print("5. Select 'USB' mode in the app")
        return "direct_usb", None
    
    elif choice == "2":
        print("\n EpocCam WiFi Setup:")
        print("1. Install 'EpocCam Webcamera' from App Store")
        print("2. Connect both iPhone and PC to same WiFi")
        print("3. Open EpocCam app and note the IP address")
        ip = input("Enter iPhone IP address: ").strip()
        return "epoccam", ip
    
    elif choice == "3":
        print("\n Camo Setup:")
        print("1. Install 'Camo' from App Store")
        print("2. Install Camo companion app on PC from reincubate.com")
        print("3. Connect iPhone via USB or WiFi")
        return "camo", None
    
    elif choice == "4":
        print("\n iVCam Setup:")
        print("1. Install 'iVCam Webcam' from App Store")
        print("2. Install iVCam PC client from e2esoft.com")
        print("3. Connect both devices to same WiFi")
        ip = input("Enter iPhone IP address: ").strip()
        return "ivcam", ip
    
    else:
        return "direct_usb", None  # Default

def main():
    print(" iPhone YOLOv8 Classroom Detection")
    print("=" * 50)
    
    # Setup iPhone connection
    connection_method, ip_address = setup_iphone_connection()
    
    # Load YOLOv8 model
    model_path = "results/train/handraise_write_read_detection/weights/best.pt"
    logging.info(f"Loading YOLOv8 model: {model_path}")
    
    try:
        model = YOLO(model_path)
        device = 'cuda' if model.device.type == 'cuda' else 'cpu'
        logging.info(f"Model loaded on device: {device}")
    except Exception as e:
        logging.error(f" Failed to load model: {e}")
        return
    
    # Connect to iPhone camera
    cap = connect_iphone_camera(connection_method, ip_address)
    if cap is None:
        print("\n iPhone camera connection failed!")
        print("\n Troubleshooting:")
        print("- Make sure iPhone app is running")
        print("- Check USB cable connection")
        print("- Verify both devices on same WiFi")
        print("- Try restarting the iPhone app")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    logging.info(f" iPhone camera: {width}x{height} @ {fps}FPS")
    
    # Class names for classroom behavior detection
    class_names = {0: 'handraise', 1: 'read', 2: 'write'}
    
    logging.info(" Starting iPhone YOLOv8 deployment...")
    logging.info("Press 'q' to quit")
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            # Read frame from iPhone camera
            ret, frame = cap.read()
            if not ret:
                logging.warning("⚠️ Failed to read frame from iPhone")
                continue
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                # YOLOv8 inference
                start_time = time.time()
                results = model(frame, conf=0.5, verbose=False)
                inference_time = time.time() - start_time
                
                # Process detections
                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Extract box coordinates and confidence
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            if class_id in class_names:
                                class_name = class_names[class_id]
                                detections.append({
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                                })
                                
                                # Draw bounding box with iPhone-optimized colors
                                color = (0, 255, 0) if class_name == 'handraise' else \
                                       (255, 0, 0) if class_name == 'write' else (0, 0, 255)
                                
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                                
                                # Draw label with background
                                label = f"{class_name}: {confidence:.2f}"
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(frame, (int(x1), int(y1)-h-10), (int(x1)+w, int(y1)), color, -1)
                                cv2.putText(frame, label, (int(x1), int(y1)-5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Log detection results
                detection_count += len(detections)
                log_msg = f"Frame {frame_count}: {len(detections)} detections, inference time: {inference_time:.3f}s"
                
                if detections:
                    detected_classes = [f"{d['class']}({d['confidence']:.2f})" for d in detections]
                    log_msg += f" - Detected: {detected_classes}"
                    logging.info(f" {log_msg}")
                else:
                    logging.info(log_msg)
            
            # Add iPhone camera info overlay
            info_text = f"iPhone Camera | Frame: {frame_count} | Total Detections: {detection_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Add detection instructions
            if detection_count == 0 and frame_count > 30:
                instructions = [
                    "Try these actions to test detection:",
                    " Raise your hand up high",
                    " Simulate writing with pen",
                    " Hold and read a book/paper"
                ]
                for i, instruction in enumerate(instructions):
                    y_pos = height - 100 + (i * 25)
                    cv2.putText(frame, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLOv8 Classroom Detection - iPhone Camera', frame)
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logging.info("  Detection stopped by user")
    except Exception as e:
        logging.error(f"  Error during detection: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        logging.info(" Final Statistics:")
        logging.info(f"   Total frames processed: {frame_count}")
        logging.info(f"   Total detections: {detection_count}")
        if detection_count > 0:
            logging.info(f"   Detection rate: {detection_count/max(frame_count,1)*100:.1f}%")
        logging.info("  iPhone camera deployment ended")

if __name__ == "__main__":
    main()
