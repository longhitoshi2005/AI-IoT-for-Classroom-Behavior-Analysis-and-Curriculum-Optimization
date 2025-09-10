#!/usr/bin/env python3
"""
Simple Camo + iPhone Test Script
===============================
Quick test to check if Camo is working properly before running detection.
"""

import cv2
import time

def test_camo_camera():
    print("Testing Camo + iPhone Connection")
    print("=" * 40)
    
    # Test different camera IDs
    for camera_id in [0, 1, 2]:
        print(f"\nTesting Camera ID: {camera_id}")
        print("-" * 20)
        
        # Test different backends
        backends = [
            (cv2.CAP_ANY, "Any Available"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
        ]
        
        for backend, backend_name in backends:
            print(f"Trying {backend_name}...")
            
            try:
                if backend == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(camera_id)
                else:
                    cap = cv2.VideoCapture(camera_id, backend)
                
                if not cap.isOpened():
                    print(f"  Cannot open camera")
                    continue
                
                # Wait for Camo initialization
                time.sleep(1.0)
                
                # Test frame reading
                for attempt in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"  SUCCESS! Frame shape: {frame.shape}")
                        
                        # Show live preview
                        print(f"  Starting preview (Press 'q' to quit, 'n' for next)")
                        
                        cv2.namedWindow('Camo Test', cv2.WINDOW_AUTOSIZE)
                        frame_count = 0
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                print(f"  Frame read failed at {frame_count}")
                                break
                            
                            frame_count += 1
                            
                            # Add info text
                            text = f"Camera {camera_id} - {backend_name} - Frame {frame_count}"
                            cv2.putText(frame, text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame, "Press 'q' to quit, 'n' for next", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
                            cv2.imshow('Camo Test', frame)
                            
                            key = cv2.waitKey(30) & 0xFF
                            if key == ord('q'):
                                cap.release()
                                cv2.destroyAllWindows()
                                print(f"\nFOUND WORKING SETUP!")
                                print(f"Use: python deploy_video_detection.py --input {camera_id}")
                                return True
                            elif key == ord('n'):
                                cap.release()
                                cv2.destroyAllWindows()
                                break
                        
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    
                    time.sleep(0.3)
                else:
                    print(f"  Cannot read frames")
                    cap.release()
                    
            except Exception as e:
                print(f"  Error: {e}")
                if 'cap' in locals():
                    cap.release()
    
    print(f"\nNo working camera found!")
    print(f"Troubleshooting:")
    print(f"1. Make sure Camo is running on both iPhone and PC")
    print(f"2. Check connection (WiFi/USB)")
    print(f"3. Restart Camo apps")
    print(f"4. Close other camera apps")
    return False

def main():
    print("Camo + iPhone Camera Test")
    print("Before running the main detection script, let's test your setup...")
    print()
    
    test_camo_camera()

if __name__ == "__main__":
    main()
