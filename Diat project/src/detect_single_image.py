
# -------------------------------------------
import cv2
import os
from datetime import datetime
from ultralytics import YOLO

# Load the model
model = YOLO(r'C:\Users\lenovo\OneDrive\Desktop\Diat project\runs\detect\train16\weights\best.pt')
print("Model loaded successfully!")

# Define the image path and output directory
image_path = r'C:\Users\lenovo\OneDrive\Desktop\Diat project\dataset\blade_damage_dataset\train\images\frame_82.jpg'  # replace with actual image path
# image_path = r'C:\Users\lenovo\OneDrive\Documents\may3.jpg'  # replace with actual image path

output_dir = r'C:\Users\lenovo\OneDrive\Desktop\Diat project\outputs\detection_results'  # replace with actual output directory

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Run the detection
results = model(image_path)

# Access the first result and plot it
annotated_image = results[0].plot()  # This generates an annotated image

# Generate a unique filename with the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"annotated_image_{timestamp}.jpg")

# Save the annotated image manually using OpenCV
cv2.imwrite(output_path, annotated_image)
print(f"Results saved to {output_path}")
