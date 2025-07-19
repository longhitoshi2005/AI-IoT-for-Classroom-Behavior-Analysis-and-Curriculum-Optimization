#!/usr/bin/env python3
"""
YOLOv8 Classroom Behavior Detection - Camo iPhone Camera
Optimized for Reincubate Camo professional iPhone camera app
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('camo_detection_log.txt'),
        logging.StreamHandler()
    ]
)

def find_camo_camera():
    """
    Auto-detect Camo camera from available video sources
    Camo usually appears as a USB camera device
    """
    logging.info("🔍 Searching for Camo camera...")
    
    # Try different camera indices
    for camera_index in range(10):  # Check first 10 camera indices
        logging.info(f"Testing camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            # Test if we can actually read frames
            ret, frame = cap.read()
            if ret and frame is not None:
                # Get camera properties to identify Camo
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                logging.info(f"📹 Found camera at index {camera_index}: {width}x{height} @ {fps}FPS")
                
                # Camo typically provides high-quality video
                if width >= 640 and height >= 480:
                    logging.info(f"✅ Using camera index {camera_index} (likely Camo)")
                    return cap, camera_index
            
            cap.release()
    
    logging.error("❌ No suitable camera found")
    return None, -1

def optimize_camo_settings(cap):
    """
    Optimize camera settings for Camo
    """
    logging.info("⚙️ Optimizing Camo camera settings...")
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try to set optimal resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    logging.info(f"📐 Final camera settings: {width}x{height} @ {fps}FPS")
    return width, height, fps

def main():
    print("🍎 Camo iPhone Camera YOLOv8 Detection")
    print("=" * 50)
    print("Make sure:")
    print("✓ Camo app is installed on iPhone")
    print("✓ Camo for Windows is installed on PC")
    print("✓ iPhone is connected via USB cable")
    print("✓ Camo app is running on iPhone")
    print("✓ Camo shows 'Connected' status")
    print()
    
    input("Press Enter when Camo is ready and connected...")
    
    # Load YOLOv8 model
    model_path = "results/train/handraise_write_read_detection/weights/best.pt"
    logging.info(f"🤖 Loading YOLOv8 model: {model_path}")
    
    try:
        model = YOLO(model_path)
        device = 'cuda' if model.device.type == 'cuda' else 'cpu'
        logging.info(f"🔥 Model loaded on device: {device}")
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        return
    
    # Find and connect to Camo camera
    cap, camera_index = find_camo_camera()
    if cap is None:
        print("\n❌ Camo camera not found!")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Camo app is running on iPhone")
        print("2. Check USB cable connection")
        print("3. Restart Camo app on iPhone")
        print("4. Try disconnecting and reconnecting iPhone")
        print("5. Make sure Camo for Windows is installed")
        return
    
    # Optimize camera settings
    width, height, fps = optimize_camo_settings(cap)
    
    # Class names for classroom behavior detection
    class_names = {0: 'handraise', 1: 'read', 2: 'write'}
    class_colors = {
        'handraise': (0, 255, 0),    # Green
        'read': (255, 0, 0),         # Blue  
        'write': (0, 0, 255)         # Red
    }
    
    logging.info("🚀 Starting Camo iPhone YOLOv8 deployment...")
    logging.info("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    detection_count = 0
    total_inference_time = 0
    
    try:
        while True:
            # Read frame from Camo camera
            ret, frame = cap.read()
            if not ret:
                logging.warning("⚠️ Failed to read frame from Camo camera")
                continue
            
            # YOLOv8 inference on every frame (Camo can handle it)
            start_time = time.time()
            results = model(frame, conf=0.4, verbose=False)  # Lower confidence for better detection
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
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
                            
                            # Draw professional-looking bounding box
                            color = class_colors[class_name]
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                            
                            # Draw label with background
                            label = f"{class_name.upper()}: {confidence:.2f}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            
                            # Label background
                            cv2.rectangle(frame, (int(x1), int(y1)-label_h-15), 
                                        (int(x1)+label_w+10, int(y1)), color, -1)
                            
                            # Label text
                            cv2.putText(frame, label, (int(x1)+5, int(y1)-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Update detection count
            detection_count += len(detections)
            
            # Log detection results every 30 frames to avoid spam
            if frame_count % 30 == 0 or len(detections) > 0:
                avg_fps = frame_count / max(total_inference_time, 0.001)
                log_msg = f"Frame {frame_count}: {len(detections)} detections, {avg_fps:.1f} FPS"
                
                if detections:
                    detected_classes = [f"{d['class']}({d['confidence']:.2f})" for d in detections]
                    log_msg += f" - Detected: {detected_classes}"
                    logging.info(f"✅ {log_msg}")
                elif frame_count % 30 == 0:
                    logging.info(log_msg)
            
            # Professional overlay design
            overlay_height = 80
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Camera info
            info_text = f"📱 Camo iPhone Camera | Resolution: {width}x{height}"
            cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detection stats
            avg_fps = frame_count / max(total_inference_time, 0.001)
            stats_text = f"🎯 Frame: {frame_count} | Detections: {detection_count} | FPS: {avg_fps:.1f}"
            cv2.putText(frame, stats_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Instructions overlay (show for first 5 seconds)
            if frame_count < 150:  # Show for ~5 seconds at 30fps
                instructions = [
                    "Test these classroom behaviors:",
                    "✋ Raise your hand high",
                    "✏️  Simulate writing motion", 
                    "📖 Hold and read a book"
                ]
                
                for i, instruction in enumerate(instructions):
                    y_pos = height - 120 + (i * 25)
                    # Add shadow effect
                    cv2.putText(frame, instruction, (12, y_pos+2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLOv8 Classroom Detection - Camo iPhone Camera', frame)
            
            frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"camo_detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame)
                logging.info(f"📸 Screenshot saved: {screenshot_path}")
                
    except KeyboardInterrupt:
        logging.info("🛑 Detection stopped by user")
    except Exception as e:
        logging.error(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final comprehensive statistics
        avg_fps = frame_count / max(total_inference_time, 0.001)
        avg_inference = total_inference_time / max(frame_count, 1) * 1000  # ms
        
        logging.info("📊 Final Performance Statistics:")
        logging.info(f"   Camera: Camo iPhone (Index {camera_index})")
        logging.info(f"   Resolution: {width}x{height}")
        logging.info(f"   Total frames processed: {frame_count}")
        logging.info(f"   Total detections: {detection_count}")
        logging.info(f"   Average FPS: {avg_fps:.1f}")
        logging.info(f"   Average inference time: {avg_inference:.1f}ms")
        
        if detection_count > 0:
            detection_rate = detection_count / max(frame_count, 1) * 100
            logging.info(f"   Detection rate: {detection_rate:.1f}%")
            logging.info("   Most detected behaviors in this session:")
        
        logging.info("🔚 Camo iPhone camera deployment ended")
        print(f"\n✅ Session completed! Check 'camo_detection_log.txt' for detailed logs.")

if __name__ == "__main__":
    main()
