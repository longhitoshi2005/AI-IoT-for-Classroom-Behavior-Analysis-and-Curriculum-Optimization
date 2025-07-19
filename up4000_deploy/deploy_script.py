import cv2
import numpy as np
import time
import json
import os
from openvino.runtime import Core

class UP4000Deployment:
    def __init__(self, model_path="up4000_deploy/openvino_ir/best.xml", config_path=None):
        self.model_path = model_path
        
        # Load configuration optimized for UP4000
        self.config = self.load_up4000_config(config_path)
        
        # Initialize OpenVINO model
        self.setup_openvino_model()
        
        # Initialize camera
        self.setup_camera()
        
        # Class names for 3 classes
        self.class_names = {
            0: 'handraise',
            1: 'write', 
            2: 'read'
        }
    
    def load_up4000_config(self, config_path):
        """Load UP4000 optimized configuration"""
        up4000_config = {
            'device': 'CPU',  # UP4000 uses Intel CPU
            'confidence_threshold': 0.5,
            'input_size': [640, 640],
            'camera_id': 0,
            'save_results': True,
            'results_path': 'results/',
            'num_threads': 4,  # UP4000 has 4 cores
            'performance_hint': 'THROUGHPUT'  # Optimize for edge device
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            for key, value in up4000_config.items():
                if key not in config:
                    config[key] = value
        else:
            config = up4000_config
        
        return config
    
    def setup_openvino_model(self):
        """Initialize OpenVINO model for UP4000"""
        print(f"Loading OpenVINO model for UP4000: {self.model_path}")
        
        try:
            # Initialize OpenVINO Core
            self.core = Core()
            
            # Read model
            self.model = self.core.read_model(model=self.model_path)
            
            # Compile model for UP4000 CPU
            self.compiled_model = self.core.compile_model(
                self.model, 
                device_name="CPU",
                config={
                    "PERFORMANCE_HINT": self.config['performance_hint'],
                    "CPU_THREADS_NUM": str(self.config['num_threads'])
                }
            )
            
            # Get input and output layers
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            
            print("OpenVINO model loaded successfully on UP4000")
            print(f"Input shape: {self.input_layer.shape}")
            print(f"Output shape: {self.output_layer.shape}")
            
        except Exception as e:
            print(f"Error loading OpenVINO model: {e}")
            print("Make sure you have converted your model to OpenVINO format:")
            print("Run: yolo export model=best.pt format=openvino")
            raise
    
    def setup_camera(self):
        """Initialize camera for UP4000"""
        print("Initializing camera for UP4000...")
        
        # Try different camera indices for UP4000
        camera_indices = [0, 1, 2]  # UP4000 might use different indices
        
        for cam_id in camera_indices:
            self.cap = cv2.VideoCapture(cam_id)
            if self.cap.isOpened():
                print(f"Camera found at index {cam_id}")
                self.config['camera_id'] = cam_id
                break
        
        if not self.cap.isOpened():
            raise ValueError("No camera found. Check camera connection to UP4000")
        
        # Set camera properties optimized for UP4000
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution for UP4000
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)            # Conservative FPS for UP4000
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer lag
        
        print("Camera initialized for UP4000")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for OpenVINO inference"""
        # Resize to model input size
        input_height, input_width = self.config['input_size']
        resized = cv2.resize(frame, (input_width, input_height))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to NCHW
        normalized = rgb_frame.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor
    
    def postprocess_output(self, output, orig_height, orig_width):
        """Postprocess OpenVINO output"""
        detections = []
        
        # OpenVINO YOLOv8 output format: [batch, 84, 8400]
        # 84 = 4 (box coords) + 80 (classes) for COCO, but we have 3 classes
        # So it should be [batch, 7, 8400] = 4 (box) + 3 (our classes)
        
        output = output[0]  # Remove batch dimension
        
        # Transpose to [8400, 7]
        if len(output.shape) == 2:
            output = output.T
        
        for detection in output:
            # Extract box coordinates and scores
            if len(detection) >= 7:  # Make sure we have enough elements
                x_center, y_center, width, height = detection[0:4]
                class_scores = detection[4:7]  # Our 3 classes
                
                # Get best class
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if confidence >= self.config['confidence_threshold']:
                    # Convert from center format to corner format
                    x1 = int((x_center - width/2) * orig_width / self.config['input_size'][0])
                    y1 = int((y_center - height/2) * orig_height / self.config['input_size'][1])
                    x2 = int((x_center + width/2) * orig_width / self.config['input_size'][0])
                    y2 = int((y_center + height/2) * orig_height / self.config['input_size'][1])
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, f'Class_{class_id}')
                    })
        
        return detections
    
    def run_inference(self, frame):
        """Run OpenVINO inference on UP4000"""
        start_time = time.time()
        
        orig_height, orig_width = frame.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_frame(frame)
        
        # Run inference
        output = self.compiled_model([input_tensor])[self.output_layer]
        
        # Postprocess
        detections = self.postprocess_output(output, orig_height, orig_width)
        
        inference_time = time.time() - start_time
        
        return detections, inference_time
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes optimized for UP4000 display"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Colors for UP4000 display
            colors = {
                'handraise': (0, 255, 0),    # Green
                'write': (255, 0, 0),        # Blue  
                'read': (0, 0, 255)          # Red
            }
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw thicker lines for UP4000 display
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Larger text for UP4000
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
    
    def run_up4000_deployment(self):
        """Run deployment optimized for UP4000"""
        print("Starting UP4000 Student Behavior Detection...")
        print("Optimized for Intel UP4000 with OpenVINO")
        print("Press 'q' to quit")
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        total_inference_time = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Run inference
            detections, inference_time = self.run_inference(frame)
            total_inference_time += inference_time
            
            # Draw results
            frame = self.draw_detections(frame, detections)
            
            # Display performance metrics
            avg_inference = total_inference_time / (frame_count + 1)
            cv2.putText(frame, f"UP4000 Inference: {inference_time:.3f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Avg: {avg_inference:.3f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                
                cv2.putText(frame, f"UP4000 FPS: {fps:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show detection count
            cv2.putText(frame, f"Detections: {len(detections)}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('UP4000 Student Behavior Detection', frame)
            
            frame_count += 1
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print final statistics
        print(f"\nUP4000 Performance Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Average inference time: {total_inference_time/frame_count:.3f}s")
        print(f"Average FPS: {frame_count/total_inference_time:.1f}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("UP4000 deployment stopped")

if __name__ == "__main__":
    # UP4000 deployment
    model_path = "up4000_deploy/openvino_ir/best.xml"
    
    # Check if OpenVINO model exists
    if not os.path.exists(model_path):
        print(f"OpenVINO model not found at {model_path}")
        print("\nTo create OpenVINO model for UP4000:")
        print("1. First train your model: python src/models/train_model.py")
        print("2. Convert to OpenVINO: yolo export model=results/train/handraise_write_read_detection/weights/best.pt format=openvino")
        print("3. Move the exported files to up4000_deploy/openvino_ir/")
        exit(1)
    
    try:
        deployment = UP4000Deployment(model_path)
        deployment.run_up4000_deployment()
    except Exception as e:
        print(f"UP4000 deployment failed: {e}")
        print("\nTroubleshooting:")
        print("- Make sure OpenVINO is installed: pip install openvino")
        print("- Check camera connection to UP4000")
        print("- Verify OpenVINO model files exist")