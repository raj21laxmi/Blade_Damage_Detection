
from ultralytics import YOLO

# Load the YOLOv8 model (adjust this based on your needs)
model = YOLO('yolov8n.pt')  # You can use other versions like yolov8s.pt, etc.

# Run the training using the model's .train() method, but without 'hyp'
model.train(
    data='dataset.yaml',  # Path to your dataset.yaml file
    epochs=50,            # Number of epochs
    batch=16,             # Batch size
    imgsz=640,            # Image size
    name='blade_damage_detection'  # Custom name for your run
)
