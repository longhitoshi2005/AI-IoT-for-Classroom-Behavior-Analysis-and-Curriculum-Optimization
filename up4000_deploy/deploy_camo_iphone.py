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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import os

# Configure logging
# Create results/runs directory first
results_dir = os.path.join("results", "runs")
os.makedirs(results_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(results_dir, 'camo_detection_log.txt'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def find_camo_camera():
    """
    Auto-detect Camo camera from available video sources
    Camo usually appears as a USB camera device
    """
    logging.info(" Searching for Camo camera...")
    
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
                
                logging.info(f" Found camera at index {camera_index}: {width}x{height} @ {fps}FPS")
                
                # Camo typically provides high-quality video
                if width >= 640 and height >= 480:
                    logging.info(f" Using camera index {camera_index} (likely Camo)")
                    return cap, camera_index
            
            cap.release()
    
    logging.error(" No suitable camera found")
    return None, -1

def optimize_camo_settings(cap):
    """
    Optimize camera settings for Camo
    """
    logging.info(" Optimizing Camo camera settings...")
    
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
    
    logging.info(f" Final camera settings: {width}x{height} @ {fps}FPS")
    return width, height, fps

class BehaviorTracker:
    """
    Track student behaviors over time and generate analytics
    """
    def __init__(self, interval_seconds=5):
        self.interval_seconds = interval_seconds
        self.behavior_data = []
        self.current_interval_data = defaultdict(int)
        self.last_interval_time = time.time()
        self.total_counts = defaultdict(int)
        
        # Create results/runs directory for saving files
        self.results_dir = os.path.join("results", "runs")
        os.makedirs(self.results_dir, exist_ok=True)
        logging.info(f" Results will be saved to: {os.path.abspath(self.results_dir)}")
        
    def add_detection(self, detections):
        """Add detections for current frame"""
        current_time = time.time()
        
        # Count behaviors in current frame
        frame_counts = defaultdict(int)
        for detection in detections:
            behavior = detection['class']
            frame_counts[behavior] += 1
            self.total_counts[behavior] += 1
        
        # Add to current interval
        for behavior, count in frame_counts.items():
            self.current_interval_data[behavior] += count
        
        # Check if interval is complete
        if current_time - self.last_interval_time >= self.interval_seconds:
            self._save_interval()
            self._reset_interval(current_time)
    
    def _save_interval(self):
        """Save current interval data"""
        timestamp = datetime.fromtimestamp(self.last_interval_time)
        interval_data = {
            'timestamp': timestamp,
            'handraise': self.current_interval_data.get('handraise', 0),
            'write': self.current_interval_data.get('write', 0),
            'read': self.current_interval_data.get('read', 0),
            'total': sum(self.current_interval_data.values())
        }
        self.behavior_data.append(interval_data)
        
        # Log interval summary
        if interval_data['total'] > 0:
            logging.info(f" Interval Summary: {interval_data}")
    
    def _reset_interval(self, current_time):
        """Reset interval data"""
        self.current_interval_data = defaultdict(int)
        self.last_interval_time = current_time
    
    def export_to_csv(self, filename=None):
        """Export behavior data to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"behavior_timeline_{timestamp}.csv"
        
        # Save to results/runs directory
        full_path = os.path.join(self.results_dir, filename)
        
        if not self.behavior_data:
            logging.warning("⚠️ No behavior data to export")
            return None
        
        df = pd.DataFrame(self.behavior_data)
        df.to_csv(full_path, index=False)
        logging.info(f" Behavior data exported to: {full_path}")
        return full_path
    
    def generate_charts(self, csv_filename=None):
        """Generate visualization charts"""
        if not self.behavior_data:
            logging.warning("⚠️ No data available for charts")
            return
        
        df = pd.DataFrame(self.behavior_data)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Classroom Behavior Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Chart 1: Timeline of all behaviors
        ax1 = axes[0, 0]
        ax1.plot(df['timestamp'], df['handraise'], marker='o', label='Hand Raise', color='green', linewidth=2)
        ax1.plot(df['timestamp'], df['write'], marker='s', label='Writing', color='blue', linewidth=2)
        ax1.plot(df['timestamp'], df['read'], marker='^', label='Reading', color='red', linewidth=2)
        ax1.set_title('Behavior Timeline', fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Number of Detections')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Chart 2: Total behavior counts (pie chart)
        ax2 = axes[0, 1]
        total_behaviors = [self.total_counts['handraise'], self.total_counts['write'], self.total_counts['read']]
        behavior_labels = ['Hand Raise', 'Writing', 'Reading']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        wedges, texts, autotexts = ax2.pie(total_behaviors, labels=behavior_labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Total Behavior Distribution', fontweight='bold')
        
        # Chart 3: Behavior intensity heatmap
        ax3 = axes[1, 0]
        df['time_only'] = df['timestamp'].dt.strftime('%H:%M:%S')
        heatmap_data = df[['handraise', 'write', 'read']].T
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, 
                   xticklabels=[t[::max(1, len(df)//10)] for t in df['time_only']])
        ax3.set_title('Behavior Intensity Heatmap', fontweight='bold')
        ax3.set_ylabel('Behaviors')
        ax3.set_xlabel('Time Intervals')
        
        # Chart 4: Engagement metrics
        ax4 = axes[1, 1]
        engagement_scores = []
        for _, row in df.iterrows():
            # Calculate engagement score (handraise=3, write=2, read=1)
            score = (row['handraise'] * 3) + (row['write'] * 2) + (row['read'] * 1)
            engagement_scores.append(score)
        
        ax4.bar(range(len(engagement_scores)), engagement_scores, 
               color='orange', alpha=0.7, edgecolor='darkorange')
        ax4.set_title('Student Engagement Score Over Time', fontweight='bold')
        ax4.set_xlabel('Time Interval')
        ax4.set_ylabel('Engagement Score')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save charts to results/runs directory
        chart_filename = f"behavior_charts_{timestamp_str}.png"
        chart_full_path = os.path.join(self.results_dir, chart_filename)
        plt.savefig(chart_full_path, dpi=300, bbox_inches='tight')
        logging.info(f"📊 Charts saved to: {chart_full_path}")
        
        # Show charts
        plt.show()
        
        # Generate summary statistics
        self._generate_summary_report(df, timestamp_str)
        
        return chart_full_path
    
    def _generate_summary_report(self, df, timestamp_str):
        """Generate a summary report"""
        report_filename = f"behavior_report_{timestamp_str}.txt"
        report_full_path = os.path.join(self.results_dir, report_filename)
        
        with open(report_full_path, 'w') as f:
            f.write("CLASSROOM BEHAVIOR ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Session info
            f.write(f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {len(df)} intervals ({len(df) * self.interval_seconds} seconds)\n")
            f.write(f"Interval Length: {self.interval_seconds} seconds\n\n")
            
            # Total counts
            f.write("TOTAL BEHAVIOR COUNTS:\n")
            f.write(f"  Hand Raising: {self.total_counts['handraise']}\n")
            f.write(f"  Writing: {self.total_counts['write']}\n")
            f.write(f"  Reading: {self.total_counts['read']}\n")
            f.write(f"  Total Detections: {sum(self.total_counts.values())}\n\n")
            
            # Peak activity periods
            if not df.empty:
                peak_handraise = df.loc[df['handraise'].idxmax()]
                peak_write = df.loc[df['write'].idxmax()]
                peak_read = df.loc[df['read'].idxmax()]
                
                f.write("PEAK ACTIVITY PERIODS:\n")
                f.write(f"  Most Hand Raising: {peak_handraise['timestamp']} ({peak_handraise['handraise']} detections)\n")
                f.write(f"  Most Writing: {peak_write['timestamp']} ({peak_write['write']} detections)\n")
                f.write(f"  Most Reading: {peak_read['timestamp']} ({peak_read['read']} detections)\n\n")
                
                # Averages
                f.write("AVERAGE BEHAVIORS PER INTERVAL:\n")
                f.write(f"  Hand Raising: {df['handraise'].mean():.2f}\n")
                f.write(f"  Writing: {df['write'].mean():.2f}\n")
                f.write(f"  Reading: {df['read'].mean():.2f}\n")
        
        logging.info(f"📝 Summary report saved to: {report_full_path}")

def main():
    print(" Camo iPhone Camera YOLOv8 Detection")
    print("=" * 50)
    print("Make sure:")
    print("✓ Camo app is installed on iPhone")
    print("✓ Camo for Windows is installed on PC")
    print("✓ iPhone is connected via USB cable")
    print("✓ Camo app is running on iPhone")
    print("✓ Camo shows 'Connected' status")
    print()
    
    # Check if OpenCV GUI is available
    gui_available = True
    try:
        # Force GUI mode for testing
        cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("test")
        print(" OpenCV GUI is available!")
    except cv2.error as e:
        gui_available = False
        print("  OpenCV GUI not available - running in headless mode")
        print(f"GUI Error: {e}")
        print(" Will save screenshots and generate analytics only")
        print("\n To fix GUI issues, try:")
        print("1. pip uninstall opencv-python -y")
        print("2. pip install opencv-python==4.8.1.78")
    
    input("Press Enter when Camo is ready and connected...")
    
    # Load YOLOv8 model
    model_path = "results/train/handraise_write_read_detection/weights/best.pt"
    logging.info(f" Loading YOLOv8 model: {model_path}")
    
    try:
        model = YOLO(model_path)
        device = 'cuda' if model.device.type == 'cuda' else 'cpu'
        logging.info(f" Model loaded on device: {device}")
    except Exception as e:
        logging.error(f" Failed to load model: {e}")
        return
    
    # Find and connect to Camo camera
    cap, camera_index = find_camo_camera()
    if cap is None:
        print("\n Camo camera not found!")
        print("\n Troubleshooting:")
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
    
    logging.info(" Starting Camo iPhone YOLOv8 deployment...")
    if gui_available:
        logging.info("Press 'q' to quit, 's' to save screenshot, 'r' to generate report")
    else:
        logging.info("Running in HEADLESS mode - Press 'q' or 'r' keys in terminal to control")
        print(" Headless Controls:")
        print("   • Press 'q' to quit")
        print("   • Press 'r' to generate report")
        print("   • Auto-screenshots every 10 seconds")
    
    # Initialize behavior tracker
    behavior_tracker = BehaviorTracker(interval_seconds=5)  # 5-second intervals
    logging.info(" Behavior tracking initialized (5-second intervals)")
    
    frame_count = 0
    detection_count = 0
    total_inference_time = 0
    
    try:
        while True:
            # Read frame from Camo camera
            ret, frame = cap.read()
            if not ret:
                logging.warning(" Failed to read frame from Camo camera")
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
            
            # Track behaviors for analytics
            behavior_tracker.add_detection(detections)
            
            # Log detection results every 30 frames to avoid spam
            if frame_count % 30 == 0 or len(detections) > 0:
                avg_fps = frame_count / max(total_inference_time, 0.001)
                log_msg = f"Frame {frame_count}: {len(detections)} detections, {avg_fps:.1f} FPS"
                
                if detections:
                    detected_classes = [f"{d['class']}({d['confidence']:.2f})" for d in detections]
                    log_msg += f" - Detected: {detected_classes}"
                    logging.info(f" {log_msg}")
                elif frame_count % 30 == 0:
                    logging.info(log_msg)
            
            # Professional overlay design
            overlay_height = 80
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Camera info
            info_text = f" Camo iPhone Camera | Resolution: {width}x{height}"
            cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detection stats
            avg_fps = frame_count / max(total_inference_time, 0.001)
            stats_text = f" Frame: {frame_count} | Detections: {detection_count} | FPS: {avg_fps:.1f}"
            cv2.putText(frame, stats_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Instructions overlay (show for first 5 seconds)
            if frame_count < 150:  # Show for ~5 seconds at 30fps
                instructions = [
                    "Test these classroom behaviors:",
                    " Raise your hand high",
                    " Simulate writing motion", 
                    " Hold and read a book"
                ]
                
                for i, instruction in enumerate(instructions):
                    y_pos = height - 120 + (i * 25)
                    # Add shadow effect
                    cv2.putText(frame, instruction, (12, y_pos+2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display frame (only if GUI is available)
            if gui_available:
                try:
                    cv2.imshow('YOLOv8 Classroom Detection - Camo iPhone Camera', frame)
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save screenshot to results/runs directory
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_filename = f"camo_detection_screenshot_{timestamp}.jpg"
                        screenshot_path = os.path.join(behavior_tracker.results_dir, screenshot_filename)
                        cv2.imwrite(screenshot_path, frame)
                        logging.info(f" Screenshot saved: {screenshot_path}")
                    elif key == ord('r'):
                        # Generate report immediately
                        logging.info(" Generating behavior report...")
                        csv_file = behavior_tracker.export_to_csv()
                        if csv_file:
                            behavior_tracker.generate_charts(csv_file)
                            logging.info(" Report generated successfully!")
                        else:
                            logging.warning(" No data available for report generation")
                except cv2.error as gui_error:
                    logging.warning(f"GUI display error: {gui_error}")
                    gui_available = False
                    print(" Switched to headless mode due to GUI error")
            else:
                # Headless mode - auto-save screenshots periodically
                if frame_count % 300 == 0 and frame_count > 0:  # Every 10 seconds at 30fps
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_filename = f"auto_screenshot_{timestamp}.jpg"
                    screenshot_path = os.path.join(behavior_tracker.results_dir, screenshot_filename)
                    cv2.imwrite(screenshot_path, frame)
                    logging.info(f" Auto-screenshot saved: {screenshot_path}")
                
                # Check for exit condition (you can modify this)
                import msvcrt
                if msvcrt.kbhit():
                    key = ord(msvcrt.getch())
                    if key == ord('q') or key == ord('Q'):
                        print(" Exit requested via keyboard")
                        break
                    elif key == ord('r') or key == ord('R'):
                        # Generate report immediately
                        logging.info(" Generating behavior report...")
                        csv_file = behavior_tracker.export_to_csv()
                        if csv_file:
                            behavior_tracker.generate_charts(csv_file)
                            logging.info(" Report generated successfully!")
                        else:
                            logging.warning(" No data available for report generation")
            
            frame_count += 1
                
    except KeyboardInterrupt:
        logging.info(" Detection stopped by user")
    except Exception as e:
        logging.error(f" Error during detection: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cap.release()
        if gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore cleanup errors
        
        # Generate final behavior analytics
        logging.info(" Generating final behavior analytics...")
        csv_file = behavior_tracker.export_to_csv()
        if csv_file:
            chart_file = behavior_tracker.generate_charts(csv_file)
            logging.info(" Final analytics generated successfully!")
        
        # Final comprehensive statistics
        avg_fps = frame_count / max(total_inference_time, 0.001)
        avg_inference = total_inference_time / max(frame_count, 1) * 1000  # ms
        
        logging.info(" Final Performance Statistics:")
        logging.info(f"   Camera: Camo iPhone (Index {camera_index})")
        logging.info(f"   Resolution: {width}x{height}")
        logging.info(f"   Total frames processed: {frame_count}")
        logging.info(f"   Total detections: {detection_count}")
        logging.info(f"   Average FPS: {avg_fps:.1f}")
        logging.info(f"   Average inference time: {avg_inference:.1f}ms")
        
        # Behavior analytics summary
        if hasattr(behavior_tracker, 'total_counts') and sum(behavior_tracker.total_counts.values()) > 0:
            logging.info(" Behavior Analytics Summary:")
            logging.info(f"   Total Hand Raises: {behavior_tracker.total_counts['handraise']}")
            logging.info(f"   Total Writing Activities: {behavior_tracker.total_counts['write']}")
            logging.info(f"   Total Reading Activities: {behavior_tracker.total_counts['read']}")
            
            detection_rate = detection_count / max(frame_count, 1) * 100
            logging.info(f"   Detection rate: {detection_rate:.1f}%")
            
            # Calculate engagement metrics
            total_behaviors = sum(behavior_tracker.total_counts.values())
            engagement_score = (
                behavior_tracker.total_counts['handraise'] * 3 +
                behavior_tracker.total_counts['write'] * 2 +
                behavior_tracker.total_counts['read'] * 1
            ) / max(total_behaviors, 1)
            logging.info(f"   Average Engagement Score: {engagement_score:.2f}/3.0")
        elif detection_count > 0:
            detection_rate = detection_count / max(frame_count, 1) * 100
            logging.info(f"   Detection rate: {detection_rate:.1f}%")
            logging.info("   Most detected behaviors in this session:")
        
        logging.info(" Camo iPhone camera deployment ended")
        print(f"\n Session completed!")
        print(f" All files saved to: {os.path.abspath(behavior_tracker.results_dir)}")
        print(f" Check the results/runs/ folder for:")
        print(f"   • CSV data files")
        print(f"   • Behavior charts")
        print(f"   • Summary reports")
        print(f"   • Screenshots")
        print(f"   • Session logs")

if __name__ == "__main__":
    main()
