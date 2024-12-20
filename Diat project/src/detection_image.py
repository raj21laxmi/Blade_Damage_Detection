import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque
import logging
import time

class BorescopeFrameProcessor:
    def __init__(self):
        self.frame_buffer = deque(maxlen=5)  # Store recent frames
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            filename='blade_inspection.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def preprocess_frame(self, frame):
        """
        Preprocess borescope frame for damage detection
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Handle glare and reflections common in borescope footage
            glare_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
            gray = cv2.inpaint(gray, glare_mask, 3, cv2.INPAINT_TELEA)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Edge enhancement
            edges = self.enhance_edges(denoised)
            
            return edges, denoised
            
        except Exception as e:
            logging.error(f"Frame preprocessing failed: {str(e)}")
            return None, None

    def enhance_edges(self, img):
        """
        Enhanced edge detection for borescope images
        """
        # Bilateral filter to preserve edges while reducing noise
        bilateral = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Gradient calculation using Sobel
        grad_x = cv2.Sobel(bilateral, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(bilateral, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = np.uint8(gradient * 255 / gradient.max())
        
        return gradient

class BladeDetector:
    """Detect and isolate blade regions in borescope frames"""
    
    def __init__(self):
        self.blade_cascade = self.setup_cascade()
        
    def setup_cascade(self):
        # Note: You'll need to train a cascade classifier specifically for blades
        # This is a placeholder path
        try:
            cascade = cv2.CascadeClassifier('blade_cascade.xml')
            return cascade
        except:
            logging.error("Cascade classifier not found")
            return None
    
    def detect_blades(self, frame):
        """
        Detect blade regions in the frame
        Returns list of regions of interest (ROIs)
        """
        if self.blade_cascade is None:
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blades = self.blade_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return blades

class DamageDetector:
    def __init__(self):
        self.model = self.load_model()
        self.processor = BorescopeFrameProcessor()
        self.blade_detector = BladeDetector()
        
    def load_model(self):
        """
        Load pre-trained damage detection model
        """
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = nn.Linear(512, 2)  # Binary classification
        
        try:
            model.load_state_dict(torch.load('blade_damage_model.pth'))
            model.eval()
        except:
            logging.warning("No pretrained model found, using untrained model")
            
        return model

    def process_video(self, video_path, output_path=None):
        """
        Process borescope inspection video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return
            
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        frames_processed = 0
        damage_detected = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, results = self.analyze_frame(frame)
            
            if results['damage_detected']:
                damage_detected.append({
                    'frame_number': frames_processed,
                    'timestamp': frames_processed / cap.get(cv2.CAP_PROP_FPS),
                    'damage_score': results['damage_score'],
                    'location': results['damage_location']
                })
            
            # Write processed frame if output requested
            if writer is not None:
                writer.write(processed_frame)
                
            frames_processed += 1
            
        cap.release()
        if writer is not None:
            writer.release()
            
        return damage_detected

    def analyze_frame(self, frame):
        """
        Analyze a single frame for blade damage
        """
        # Detect blade regions
        blade_regions = self.blade_detector.detect_blades(frame)
        
        results = {
            'damage_detected': False,
            'damage_score': 0.0,
            'damage_location': None
        }
        
        # Process each detected blade
        for (x, y, w, h) in blade_regions:
            blade_roi = frame[y:y+h, x:x+w]
            
            # Preprocess ROI
            processed_roi, denoised = self.processor.preprocess_frame(blade_roi)
            if processed_roi is None:
                continue
                
            # Prepare for model
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(processed_roi).unsqueeze(0)
            
            # Model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output)[0]
                
                if probability[1] > 0.5:  # Damage detected
                    results['damage_detected'] = True
                    results['damage_score'] = float(probability[1])
                    results['damage_location'] = (x, y, w, h)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f'Damage: {probability[1]:.2f}', 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 0, 255), 2)
        
        return frame, results

def main():
    detector = DamageDetector()
    video_path = 'borescope_inspection.mp4'
    output_path = 'analyzed_inspection.avi'
    
    print("Starting video analysis...")
    damage_detected = detector.process_video(video_path, output_path)
    
    # Report findings
    if damage_detected:
        print("\nDamage detected in the following frames:")
        for damage in damage_detected:
            print(f"Frame {damage['frame_number']} at {damage['timestamp']:.2f}s")
            print(f"Damage confidence: {damage['damage_score']:.2f}")
            print("---")
    else:
        print("\nNo damage detected in the video.")

if __name__ == "__main__":
    main()