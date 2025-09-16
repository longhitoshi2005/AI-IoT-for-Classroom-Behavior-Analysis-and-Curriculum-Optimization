#!/usr/bin/env python3
"""
Video Deployment for Classroom Behavior Detection
================================================

Real-time detection of classroom behaviors using video input or camera:
- Class 0: handraise (hand raised)
- Class 1: studying (writing/taking notes and reading - combined)
- Class 2: phone_usage (using phone)

Features:
- Video file input support
- Real-time camera detection
- Confidence filtering
- FPS counter
- Detection logging
- Behavior analytics
- Export results
- Interactive controls

Author: GitHub Copilot
Date: August 2025
"""

import cv2
import numpy as np
import time
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import sys

class VideoClassroomDetector:
    """Video and camera based classroom behavior detection"""
    
    def __init__(self, model_path, confidence_threshold=0.5, input_source=0, mode='learning'):
        """
        Initialize the video detector
        
        Args:
            model_path: Path to the trained YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            input_source: Video file path or camera device ID (0 for default camera)
            mode: Detection mode (learning, fighting, cheating)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.input_source = input_source
        self.is_video_file = isinstance(input_source, str) and input_source != "0"
        self.mode = mode

        # Set class names and colors based on mode
        if mode == 'learning':
            self.class_names = {0: 'handraise', 1: 'studying', 2: 'phone_usage'}
            self.class_colors = {
                0: (0, 255, 0),    # Green for handraise
                1: (255, 0, 0),    # Blue for studying
                2: (0, 0, 255),    # Red for phone_usage
            }
            self.unique_behaviors = ['handraise', 'studying', 'phone_usage']
        elif mode == 'fighting':
            self.class_names = {0: 'fighting'}
            self.class_colors = {0: (0, 0, 255)}  # Red for fighting
            self.unique_behaviors = ['fighting']
        elif mode == 'cheating':
            self.class_names = {0: 'cheating', 1: 'not_cheating'}
            self.class_colors = {
                0: (0, 0, 255), # Red for cheating
                1: (0, 255, 0), # Green for not_cheating
            }
            self.unique_behaviors = ['cheating', 'not_cheating']
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Internal mapping (set after model load). If the underlying model has 4 classes
        # (handraise, write, read, phone_usage) we collapse write+read -> studying.
        # If the model already has 3 classes we keep identity mapping.
        self.collapse_mapping = None  # Will be configured dynamically
        
        # Detection tracking
        self.detections_log = []
        self.frame_count = 0
        self.start_time = time.time()
        self.total_frames = 0  # For video files
        
        # Performance tracking
        self.fps_counter = []
        
        # Behavior occurrence tracking
        self.behavior_counts = {name: 0 for name in self.unique_behaviors}
        self.current_behaviors = set()
        self.behavior_sessions = []
        self.behavior_states = {}
        self.frame_detections = []
        
        # Video playback control
        self.paused = False
        self.playback_speed = 1.0
        
        print("Video Classroom Detector Initialized")
        print(f"Model: {self.model_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Input source: {self.input_source}")
        print(f"Source type: {'Video file' if self.is_video_file else 'Camera'}")
        
        # Load model
        self.load_model()
        
        # Setup input source
        self.setup_input()
    
    def load_model(self):
        """Load the trained YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print("Loading YOLO model...")
        self.model = YOLO(str(self.model_path))
        print("Model loaded successfully!")
        self.configure_class_mapping()

        print("Deployment class schema (3-class unified):")
        for cid, name in self.class_names.items():
            print(f"  {cid}: {name}")
        if self.collapse_mapping:
            print("Underlying model had 4 classes -> collapsing write/read into studying.")
        else:
            print("Underlying model already matches 3-class schema (no collapse needed).")

    def configure_class_mapping(self):
        """Configure mapping based on underlying model's native class count."""
        try:
            native_names = self.model.names  # dict or list from ultralytics
            if isinstance(native_names, dict):
                native_class_count = len(native_names)
            else:
                native_class_count = len(list(native_names))
        except Exception:
            native_class_count = 3  # Fallback assumption
        
        if native_class_count == 4:
            # Original 4-class model: 0 handraise, 1 write, 2 read, 3 phone_usage
            self.collapse_mapping = {0: 0, 1: 1, 2: 1, 3: 2}
        elif native_class_count == 3:
            self.collapse_mapping = None  # Identity
        else:
            print(f"Warning: Unexpected native class count ({native_class_count}). Assuming first 3 map directly.")
            self.collapse_mapping = None
    
    def setup_input(self):
        """Initialize video capture from file or camera"""
        if self.is_video_file:
            self.setup_video_file()
        else:
            self.setup_camera()
    
    def setup_video_file(self):
        """Initialize video file capture"""
        print(f"Setting up video file: {self.input_source}")
        
        if not Path(self.input_source).exists():
            raise FileNotFoundError(f"Video file not found: {self.input_source}")
        
        self.cap = cv2.VideoCapture(str(self.input_source))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.input_source}")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = self.total_frames / self.original_fps if self.original_fps > 0 else 0
        
        print("Video file initialized:")
        print(f"   Resolution: {self.frame_width}x{self.frame_height}")
        print(f"   FPS: {self.original_fps}")
        print(f"   Total frames: {self.total_frames}")
        print(f"   Duration: {duration_seconds:.1f} seconds")
        
        # Test frame reading
        print("Testing video file reading...")
        ret, test_frame = self.cap.read()
        if ret and test_frame is not None and test_frame.size > 0:
            print("Video file reading test successful!")
            print(f"Test frame shape: {test_frame.shape}")
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            raise RuntimeError("Cannot read frames from video file")
    
    def setup_camera(self):
        """Initialize camera capture"""
        camera_id = int(self.input_source) if str(self.input_source).isdigit() else 0
        print(f"Setting up camera {camera_id}...")
        print("Optimized for Camo + iPhone setup...")
        
        # Try multiple backends - optimized for Camo/virtual cameras
        backends_to_try = [
            (cv2.CAP_ANY, "Any Available"),      # Best for Camo
            (cv2.CAP_DSHOW, "DirectShow"),      # Windows DirectShow
            (cv2.CAP_MSMF, "Media Foundation"), # Windows Media Foundation
        ]
        
        self.cap = None
        success_backend = None
        
        for backend, backend_name in backends_to_try:
            print(f"Trying {backend_name} backend...")
            
            try:
                if backend == cv2.CAP_ANY:
                    test_cap = cv2.VideoCapture(camera_id)
                else:
                    test_cap = cv2.VideoCapture(camera_id, backend)
                
                if test_cap.isOpened():
                    # Wait longer for virtual cameras like Camo
                    import time
                    time.sleep(1.0)  # Increased wait time for Camo
                    
                    # Try multiple frame reads for stability
                    successful_read = False
                    for attempt in range(3):
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            successful_read = True
                            break
                        time.sleep(0.3)
                    
                    if successful_read:
                        self.cap = test_cap
                        success_backend = backend_name
                        print(f"Successfully connected using {backend_name}")
                        print(f"Test frame shape: {test_frame.shape}")
                        break
                    else:
                        print(f"{backend_name}: Camera opened but cannot read frames")
                        test_cap.release()
                else:
                    print(f"{backend_name}: Cannot open camera")
                    test_cap.release()
                    
            except Exception as e:
                print(f"{backend_name}: Exception occurred - {e}")
                if 'test_cap' in locals():
                    test_cap.release()
        
        if self.cap is None:
            print("\nTROUBLESHOOTING CAMO + IPHONE ISSUES:")
            print("   1. Make sure Camo app is running on computer")
            print("   2. Make sure Camo app is running on iPhone")
            print("   3. iPhone and computer should be connected (WiFi/USB)")
            print("   4. Check if Camo camera is active (green light)")
            print("   5. Close other camera applications")
            print("   6. Try different camera IDs: 0, 1, 2...")
            print("   7. Restart both Camo apps if needed")
            raise RuntimeError(f"Cannot access camera {camera_id}")
        
        # Set camera properties optimized for Camo/iPhone
        print("Configuring camera settings for Camo...")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get current resolution from Camo (don't force change)
        current_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Current Camo resolution: {current_width}x{current_height}")
        
        # Only try to set resolution if current one is invalid
        if current_width <= 0 or current_height <= 0:
            print("Invalid resolution, trying to set...")
            resolutions_to_try = [(1280, 720), (640, 480), (320, 240)]
            for width, height in resolutions_to_try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if actual_width > 0 and actual_height > 0:
                    print(f"Resolution set to: {actual_width}x{actual_height}")
                    break
        else:
            print("Using Camo's native resolution")
        
        # Don't force FPS for Camo (let it manage its own FPS)
        current_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camo FPS: {current_fps}")
        
        # Get final properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = 0  # Unknown for camera
        
        print("Camera fully initialized:")
        print(f"   Resolution: {self.frame_width}x{self.frame_height}")
        print(f"   FPS: {self.original_fps}")
        print(f"   Backend: {success_backend}")
    
    def detect_frame(self, frame):
        """Run detection on a single frame"""
        try:
            if frame is None or frame.size == 0:
                return []
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            detections = []
            frame_behaviors = set()
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        
                        if confidence >= self.confidence_threshold:
                            original_class_id = int(box.cls[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            # Apply collapse mapping if present
                            if self.collapse_mapping:
                                if original_class_id not in self.collapse_mapping:
                                    continue  # Skip unexpected class
                                new_class_id = self.collapse_mapping[original_class_id]
                            else:
                                new_class_id = original_class_id

                            if new_class_id not in self.class_names:
                                continue

                            display_name = self.class_names[new_class_id]

                            detection = {
                                'class_id': new_class_id,
                                'original_class_id': original_class_id,
                                'class_name': display_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            }
                            detections.append(detection)
                            frame_behaviors.add(display_name)
            
            # Update behavior state tracking
            self.update_behavior_states(frame_behaviors)
            
            # Store frame detections
            self.frame_detections.append({
                'frame': self.frame_count,
                'timestamp': datetime.now(),
                'behaviors': list(frame_behaviors),
                'detections': detections
            })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def update_behavior_states(self, current_frame_behaviors):
        """Update behavior state tracking for occurrence-based counting"""
        current_time = datetime.now()
        
        # Check for new behaviors (behavior started)
        new_behaviors = current_frame_behaviors - self.current_behaviors
        for behavior in new_behaviors:
            if behavior in self.unique_behaviors:
                self.behavior_states[behavior] = current_time
                self.behavior_counts[behavior] += 1
                print(f"Started: {behavior} (Total: {self.behavior_counts[behavior]})")
        
        # Check for ended behaviors (behavior stopped)
        ended_behaviors = self.current_behaviors - current_frame_behaviors
        for behavior in ended_behaviors:
            if behavior in self.behavior_states:
                start_time = self.behavior_states[behavior]
                duration = (current_time - start_time).total_seconds()
                
                session = {
                    'behavior': behavior,
                    'start_time': start_time,
                    'end_time': current_time,
                    'duration_seconds': duration,
                    'frame_start': self.frame_count,
                    'frame_end': self.frame_count
                }
                
                self.behavior_sessions.append(session)
                del self.behavior_states[behavior]
                print(f"Ended: {behavior} (Duration: {duration:.1f}s)")
        
        # Update current behaviors
        self.current_behaviors = current_frame_behaviors
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        for detection in detections:
            class_id = detection['class_id']
            class_name = detection['class_name']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return frame
    
    def draw_info_panel(self, frame):
        """Draw information panel on frame"""
        # Calculate current FPS
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Info panel background
        panel_height = 200
        cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
        
        y_pos = 30
        
        # Title
        title = "Video Behavior Detector" if self.is_video_file else "Camera Behavior Detector"
        cv2.putText(frame, title, (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25
        
        # FPS and frame info
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 20
        
        # Frame progress for video files
        if self.is_video_file and self.total_frames > 0:
            progress = (self.frame_count / self.total_frames) * 100
            cv2.putText(frame, f"Progress: {self.frame_count}/{self.total_frames} ({progress:.1f}%)", 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, f"Frames: {self.frame_count}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 20
        
        # Playback controls for video
        if self.is_video_file:
            status = "PAUSED" if self.paused else f"PLAYING ({self.playback_speed}x)"
            cv2.putText(frame, f"Status: {status}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_pos += 20
        
        # Current active behaviors
        active_behaviors = ", ".join(self.current_behaviors) if self.current_behaviors else "None"
        cv2.putText(frame, f"Active: {active_behaviors}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y_pos += 20
        
        # Behavior counts
        cv2.putText(frame, "Behavior Occurrences:", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 15
        
        for class_name, count in self.behavior_counts.items():
            color = (0, 255, 0) if class_name in self.current_behaviors else (255, 255, 255)
            cv2.putText(frame, f"  {class_name}: {count}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_pos += 15
    
    def log_detections(self, detections, timestamp):
        """Log detections to internal storage"""
        for detection in detections:
            log_entry = {
                'timestamp': timestamp,
                'frame': self.frame_count,
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            }
            self.detections_log.append(log_entry)
    
    def run_detection(self, save_video=False, output_dir="results/video_detection"):
        """Run detection loop with video controls"""
        source_type = "video file" if self.is_video_file else "camera"
        print(f"\nStarting detection from {source_type}...")
        
        if self.is_video_file:
            print("Video Controls:")
            print("  SPACE: Play/Pause")
            print("  +/-: Speed up/slow down")
            print("  LEFT/RIGHT: Skip backward/forward (10 frames)")
        
        print("General Controls:")
        print("  S: Save current results")
        print("  R: Reset counters")  
        print("  C: Take screenshot")
        print("  Q or ESC: Quit")
        print()

        # Create a unique run directory for every execution
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"run_{run_ts}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Video writer setup
        video_writer = None
        if save_video:
            video_path = output_path / f"detection_output_{run_ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, 20.0, 
                (self.frame_width, self.frame_height)
            )
            print(f"Saving output video to: {video_path}")
        
        consecutive_failures = 0
        max_failures = 10
        
        try:
            print("Starting detection loop...")
            print("Looking for OpenCV window...")
            
            # Create window explicitly to avoid display issues
            cv2.namedWindow('Classroom Behavior Detection - Video', cv2.WINDOW_AUTOSIZE)
            print("OpenCV window created")
            
            while True:
                if not self.paused or not self.is_video_file:
                    # Read frame
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        if self.is_video_file:
                            print("End of video file reached")
                            break
                        else:
                            consecutive_failures += 1
                            print(f"Frame read failed (attempt {consecutive_failures}/{max_failures})")
                            if consecutive_failures >= max_failures:
                                print("Too many consecutive failures")
                                break
                            cv2.waitKey(100)
                            continue
                    
                    consecutive_failures = 0
                    
                    # Validate frame
                    if frame is None or frame.size == 0:
                        print("Empty or invalid frame, skipping...")
                        continue
                    
                    # Debug frame info every 30 frames
                    if self.frame_count % 30 == 0:
                        print(f"Processing frame {self.frame_count}: {frame.shape}")
                    
                    self.frame_count += 1
                    timestamp = datetime.now()
                    
                    # Run detection
                    detections = self.detect_frame(frame)
                    
                    # Log detections
                    if detections:
                        self.log_detections(detections, timestamp)
                    
                    # Draw detections and info
                    frame = self.draw_detections(frame, detections)
                    self.draw_info_panel(frame)
                    
                    # Save video frame
                    if video_writer is not None:
                        video_writer.write(frame)
                
                # Display frame with error checking
                try:
                    cv2.imshow('Classroom Behavior Detection - Video', frame)
                    
                    # Debug window display every 60 frames
                    if self.frame_count % 60 == 0:
                        print(f"Window updated at frame {self.frame_count}")
                        
                except Exception as display_error:
                    print(f"Display error: {display_error}")
                    print("Trying to recreate window...")
                    cv2.destroyAllWindows()
                    cv2.namedWindow('Classroom Behavior Detection - Video', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Classroom Behavior Detection - Video', frame)
                
                # Handle key presses with improved timing
                wait_time = int(1000 / (self.original_fps * self.playback_speed)) if self.is_video_file and self.original_fps > 0 else 30
                # Ensure minimum wait time for proper window updates
                wait_time = max(1, min(wait_time, 100))
                
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    print("Stopping detection...")
                    break
                elif key == ord(' ') and self.is_video_file:  # SPACE - play/pause
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "PLAYING"
                    print(f"Video {status}")
                elif key == ord('+') and self.is_video_file:  # Speed up
                    self.playback_speed = min(4.0, self.playback_speed * 1.5)
                    print(f"Playback speed: {self.playback_speed:.1f}x")
                elif key == ord('-') and self.is_video_file:  # Slow down
                    self.playback_speed = max(0.25, self.playback_speed / 1.5)
                    print(f"Playback speed: {self.playback_speed:.1f}x")
                elif key == 83 and self.is_video_file:  # RIGHT arrow - skip forward
                    for _ in range(10):
                        ret, _ = self.cap.read()
                        if not ret:
                            break
                    print("Skipped forward 10 frames")
                elif key == 81 and self.is_video_file:  # LEFT arrow - skip backward
                    current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    new_pos = max(0, current_pos - 10)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    print("Skipped backward 10 frames")
                elif key == ord('c'):  # C - screenshot
                    screenshot_path = output_path / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(screenshot_path), frame)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord('s'):  # S - save results
                    self.save_results(output_path)
                    print("Saved CSV/JSON to:", output_path)
                elif key == ord('r'):  # R - reset counters
                    self.reset_counters()
                    print("Counters reset!")
        
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Save final results (CSV/JSON only)
            self.save_results(output_path)
            print(f"\nDetection completed!")
            print(f"Run folder: {output_path}")
            self.print_final_stats()
    
    def reset_counters(self):
        """Reset detection counters and behavior states"""
        self.behavior_counts = {name: 0 for name in self.unique_behaviors}
        self.current_behaviors = set()
        self.behavior_sessions = []
        self.behavior_states = {}
        self.detections_log = []
        self.frame_detections = []
        self.frame_count = 0
        self.start_time = time.time()
        print("All counters and behavior states reset!")
    
    def save_results(self, output_dir):
        """Save detection results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Close any ongoing behavior sessions
        self.close_ongoing_sessions()
        
        # Save behavior sessions
        if self.behavior_sessions:
            sessions_path = output_dir / f"behavior_sessions_{timestamp}.csv"
            sessions_df = pd.DataFrame(self.behavior_sessions)
            sessions_df.to_csv(sessions_path, index=False)
            print(f"Behavior sessions saved: {sessions_path}")
        
        # Save frame-level detections
        if self.frame_detections:
            frames_path = output_dir / f"frame_detections_{timestamp}.csv"
            
            frame_rows = []
            for frame_data in self.frame_detections:
                for behavior in frame_data['behaviors']:
                    frame_rows.append({
                        'frame': frame_data['frame'],
                        'timestamp': frame_data['timestamp'],
                        'behavior': behavior
                    })
            
            if frame_rows:
                frames_df = pd.DataFrame(frame_rows)
                frames_df.to_csv(frames_path, index=False)
                print(f"Frame detections saved: {frames_path}")
        
        # Save traditional detection log
        if self.detections_log:
            csv_path = output_dir / f"detections_{timestamp}.csv"
            df = pd.DataFrame(self.detections_log)
            df.to_csv(csv_path, index=False)
            print(f"Traditional detections saved: {csv_path}")
        
        # Save enhanced summary statistics
        avg_session_durations = {}
        for behavior in self.unique_behaviors:
            behavior_sessions = [s for s in self.behavior_sessions if s['behavior'] == behavior]
            if behavior_sessions:
                avg_duration = sum(s['duration_seconds'] for s in behavior_sessions) / len(behavior_sessions)
                avg_session_durations[behavior] = avg_duration
            else:
                avg_session_durations[behavior] = 0
        
        stats = {
            'session_info': {
                'timestamp': timestamp,
                'input_source': str(self.input_source),
                'source_type': 'video_file' if self.is_video_file else 'camera',
                'total_frames_processed': self.frame_count,
                'total_frames_available': self.total_frames if self.is_video_file else 'unknown',
                'duration_seconds': time.time() - self.start_time,
                'model_path': str(self.model_path),
                'confidence_threshold': self.confidence_threshold
            },
            'behavior_analysis': {
                'occurrence_counts': self.behavior_counts,
                'total_sessions': len(self.behavior_sessions),
                'average_session_durations': avg_session_durations,
                'currently_active': list(self.current_behaviors)
            },
            'legacy_stats': {
                'total_detections': len(self.detections_log)
            }
        }
        
        json_path = output_dir / f"enhanced_session_stats_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Enhanced statistics saved: {json_path}")
        
        # Print summary (CSV/JSON only)
        print(f"\nFILES CREATED:")
        print(f"=" * 40)
        print(f"Statistics: enhanced_session_stats_{timestamp}.json")
        if self.behavior_sessions:
            print(f"Sessions: behavior_sessions_{timestamp}.csv")
        if self.frame_detections:
            print(f"Frame data: frame_detections_{timestamp}.csv")
        if self.detections_log:
            print(f"Detections: detections_{timestamp}.csv")
        print(f"Location: {output_dir}")
        print(f"=" * 40)
    
    def close_ongoing_sessions(self):
        """Close any behaviors that are still active"""
        current_time = datetime.now()
        
        for behavior, start_time in list(self.behavior_states.items()):
            duration = (current_time - start_time).total_seconds()
            
            session = {
                'behavior': behavior,
                'start_time': start_time,
                'end_time': current_time,
                'duration_seconds': duration,
                'frame_start': self.frame_count,
                'frame_end': self.frame_count,
                'note': 'Session closed at end of processing'
            }
            
            self.behavior_sessions.append(session)
        
        self.behavior_states.clear()
    
    def create_enhanced_behavior_charts(self, output_dir, timestamp):
        """Charts disabled by request. Keeping CSV/JSON only."""
        print("Charts disabled: skipping PNG generation.")
    
    def print_final_stats(self):
        """Print final detection statistics"""
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print("\nENHANCED FINAL STATISTICS")
        print("=" * 50)
        print(f"Input source: {self.input_source}")
        print(f"Source type: {'Video file' if self.is_video_file else 'Camera'}")
        print(f"Total frames processed: {self.frame_count}")
        if self.is_video_file and self.total_frames > 0:
            completion_rate = (self.frame_count / self.total_frames) * 100
            print(f"Video completion: {completion_rate:.1f}% ({self.frame_count}/{self.total_frames})")
        print(f"Total processing time: {elapsed_time:.1f} seconds")
        print(f"Average FPS: {avg_fps:.1f}")
        
        print(f"\nBEHAVIOR OCCURRENCE ANALYSIS")
        print("=" * 50)
        print("Behavior occurrences (complete actions):")
        total_occurrences = sum(self.behavior_counts.values())
        for behavior, count in self.behavior_counts.items():
            percentage = (count / total_occurrences * 100) if total_occurrences > 0 else 0
            print(f"  {behavior}: {count} occurrences ({percentage:.1f}%)")
        
        print(f"\nSESSION ANALYSIS")
        print("=" * 50)
        print(f"Total behavior sessions: {len(self.behavior_sessions)}")
        
        if self.behavior_sessions:
            sessions_df = pd.DataFrame(self.behavior_sessions)
            avg_durations = sessions_df.groupby('behavior')['duration_seconds'].agg(['mean', 'count'])
            
            print("Average session durations:")
            for behavior in avg_durations.index:
                mean_duration = avg_durations.loc[behavior, 'mean']
                session_count = avg_durations.loc[behavior, 'count']
                print(f"  {behavior}: {mean_duration:.1f}s avg ({session_count} sessions)")
        
        if self.current_behaviors:
            print(f"\nBehaviors still active at end:")
            for behavior in self.current_behaviors:
                if behavior in self.behavior_states:
                    duration = (datetime.now() - self.behavior_states[behavior]).total_seconds()
                    print(f"  {behavior}: active for {duration:.1f}s")
        
        print(f"\nTRADITIONAL DETECTION COUNT: {len(self.detections_log)}")
        print()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Video/Camera deployment for classroom behavior detection')
    
    parser.add_argument('--mode', '-M',
                        choices=['learning', 'fighting', 'cheating'],
                        default='learning',
                        help='Detection mode: learning (class), fighting (break), cheating (exam)')

    parser.add_argument('--model', '-m',
                        default=None,
                        help='Path to trained model (.pt file). Overrides --mode if set.')

    parser.add_argument('--input', '-i',
                        default='0',
                        help='Input source: video file path or camera ID (default: 0 for camera)')

    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                        help='Confidence threshold (0.0-1.0)')

    parser.add_argument('--save-video', action='store_true',
                        help='Save detection output video')

    parser.add_argument('--output-dir', default='results/video_detection',
                        help='Output directory for results')

    args = parser.parse_args()

    # Map each mode to its model path (edit these paths as needed)
    MODE_MODEL_PATHS = {
        'learning': 'models/learning_behavior.pt',   # handraise, studying, phone_usage
        'fighting': 'models/fighting_behavior.pt',   # fighting
        'cheating': 'models/cheating_behavior.pt',   # cheating
    }

    # Select model path
    if args.model:
        model_path = args.model
        print(f"[MODE SWITCHER] Using custom model: {model_path}")
    else:
        model_path = MODE_MODEL_PATHS[args.mode]
        print(f"[MODE SWITCHER] Mode: {args.mode} → Model: {model_path}")

    print("Video Classroom Behavior Detection")
    print("=" * 50)
    print("CAMO + IPHONE SETUP GUIDE:")
    print("1. Install Camo app on iPhone")
    print("2. Install Camo app on Windows PC")
    print("3. Connect iPhone and PC (WiFi or USB)")
    print("4. Launch Camo on both devices")
    print("5. Wait for green light (camera active)")
    print("6. Close other camera apps")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Input: {args.input}")
    print(f"Confidence: {args.confidence}")
    print(f"Save video: {args.save_video}")
    print(f"Output: {args.output_dir}")
    
    # Check input type
    if str(args.input).isdigit():
        print(f"CAMERA MODE: Using camera ID {args.input}")
        print("TIP: Try camera IDs 0, 1, or 2 if one doesn't work")
    else:
        print(f"VIDEO MODE: Using video file {args.input}")
    
    # Check if running from terminal vs IDE
    import sys
    if hasattr(sys, 'ps1'):
        print("Running in interactive mode")
    else:
        print("Running from command line/script")
    print()
    
    try:
        print("Initializing detector...")
        
        # Initialize detector
        detector = VideoClassroomDetector(
            model_path=model_path,
            confidence_threshold=args.confidence,
            input_source=args.input,
            mode=args.mode
        )
        
        print("Starting detection...")
        
        # Run detection
        detector.run_detection(
            save_video=args.save_video,
            output_dir=args.output_dir
        )
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure the model file and input source exist")
        return 1
    
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        print("Check input source and camera connections")
        return 1
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
