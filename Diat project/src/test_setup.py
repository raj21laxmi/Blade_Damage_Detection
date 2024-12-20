# import cv2
# import torch
# import numpy as np

# def test_installations():
#     # Test OpenCV
#     print("OpenCV version:", cv2.__version__)
    
#     # Test PyTorch
#     print("PyTorch version:", torch.__version__)
#     print("CUDA available:", torch.cuda.is_available())
    
#     # Create simple test image
#     img = np.zeros((100, 100, 3), dtype=np.uint8)
#     cv2.putText(img, "Test", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
#     # Show image
#     cv2.imshow("Test Image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     test_installations()




#     blade_detection/
# ├── data/
# │   ├── train/
# │   │   ├── damaged/
# │   │   └── undamaged/
# │   └── test/
# │       ├── damaged/
# │       └── undamaged/
# ├── models/
# ├── src/
# └── venv/


import os

# Paths to each annotation directory
train_xml_dir = r"C:/Users/lenovo/OneDrive/Desktop/blade_damage_dataset/train/annotations"
val_xml_dir = r"C:/Users/lenovo/OneDrive/Desktop/blade_damage_dataset/val/annotations"
test_xml_dir = r"C:/Users/lenovo/OneDrive/Desktop/blade_damage_dataset/test/annotations"

# Output directories for YOLO labels
train_yolo_dir = r"C:/Users/lenovo/OneDrive/Desktop/blade_damage_dataset/labels/train"
val_yolo_dir = r"C:/Users/lenovo/OneDrive/Desktop/blade_damage_dataset/labels/val"
test_yolo_dir = r"C:/Users/lenovo/OneDrive/Desktop/blade_damage_dataset/labels/test"

# Make sure the directories exist
os.makedirs(train_yolo_dir, exist_ok=True)
os.makedirs(val_yolo_dir, exist_ok=True)
os.makedirs(test_yolo_dir, exist_ok=True)

# Function to convert XML to YOLO format
def convert_to_yolo_format(xml_path, output_path, class_mapping):
    # Your conversion logic here
    pass  # Replace with actual implementation

# Process each directory separately
for xml_dir, yolo_dir in zip([train_xml_dir, val_xml_dir, test_xml_dir], 
                             [train_yolo_dir, val_yolo_dir, test_yolo_dir]):
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, xml_file)
            txt_file = os.path.splitext(xml_file)[0] + ".txt"
            output_path = os.path.join(yolo_dir, txt_file)
            convert_to_yolo_format(xml_path, output_path, class_mapping)

# Class mapping (adjust based on your XML labels)
class_mapping = {
    'label1': 0,
    'label2': 1,
    # Add more labels and their corresponding indices here
}
