import cv2
import numpy as np
from openvino.runtime import Core
import time

class OpenVINOInference:
    def __init__(self, model_path, device='CPU'):
        self.core = Core()
        self.model_path = model_path
        self.device = device
        
        # Load model
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        
        # Get input/output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # FIXED: Class names for 3 classes (matching data.yaml)
        self.class_names = {
            0: 'handraise',
            1: 'write',
            2: 'read'
        }
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_layer.shape}")
        print(f"Output shape: {self.output_layer.shape}")
        print(f"Classes: {list(self.class_names.values())}")
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize to model input size
        input_h, input_w = self.input_layer.shape[2], self.input_layer.shape[3]
        resized = cv2.resize(image, (input_w, input_h))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and transpose
        input_tensor = rgb_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess(self, output, original_shape, conf_threshold=0.5):
        """Post-process YOLOv8 output"""
        detections = []
        
        # YOLOv8 output format: [batch, predictions, (x, y, w, h, conf, class0, class1, class2)]
        predictions = output[0]  # Remove batch dimension
        
        # Transpose if needed: [predictions, features] -> [features, predictions]
        if predictions.shape[0] == 3 + len(self.class_names):  # 4 bbox + 3 classes
            predictions = predictions.T
        
        for detection in predictions:
            # Extract bbox and confidence
            x_center, y_center, width, height = detection[:4]
            
            # Extract class scores (for 3 classes)
            class_scores = detection[4:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > conf_threshold:
                # Scale to original image size
                orig_h, orig_w = original_shape[:2]
                x_center *= orig_w
                y_center *= orig_h
                width *= orig_w
                height *= orig_h
                
                # Convert to bbox format
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': self.class_names.get(class_id, f'Class_{class_id}')
                })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels"""
        # FIXED: Color coding for 3 classes
        colors = {
            'handraise': (0, 255, 0),    # Green
            'write': (255, 0, 0),        # Blue  
            'read': (0, 0, 255)          # Red
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Get color for this class
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def run_inference(self, image):
        """Run inference on single image"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        start_time = time.time()
        output = self.compiled_model([input_tensor])
        inference_time = time.time() - start_time
        
        # Post-process
        detections = self.postprocess(output[self.output_layer], image.shape)
        
        return detections, inference_time

def run_webcam_inference():
    """Run real-time inference on webcam"""
    # FIXED: Correct model path
    model_path = 'up4000_deploy/openvino_ir/best.xml'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    # Initialize inference engine
    try:
        inference_engine = OpenVINOInference(model_path, device='CPU')
        print("OpenVINO inference engine initialized")
    except Exception as e:
        print(f"Failed to initialize inference engine: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("Starting webcam inference...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        detections, inference_time = inference_engine.run_inference(frame)
        
        # Draw results
        frame = inference_engine.draw_detections(frame, detections)
        
        # Show FPS and inference time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time:.3f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('OpenVINO Student Behavior Detection', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_inference()