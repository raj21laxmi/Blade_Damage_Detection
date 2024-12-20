# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import xml.etree.ElementTree as ET

def load_annotations(image_dirs):
    annotations = []
    for split_dir in image_dirs:  # 'train', 'test', 'val'
        image_dir = os.path.join(split_dir, 'images')
        annotation_dir = os.path.join(split_dir, 'annotations')
        
        for xml_file in os.listdir(annotation_dir):
            if xml_file.endswith('.xml'):
                file_path = os.path.join(annotation_dir, xml_file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                objects = []
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    obj_data = {
                        'name': obj.find('name').text,
                        'bbox': {
                            'xmin': int(bbox.find('xmin').text),
                            'ymin': int(bbox.find('ymin').text),
                            'xmax': int(bbox.find('xmax').text),
                            'ymax': int(bbox.find('ymax').text)
                        }
                    }
                    objects.append(obj_data)
                annotations.append({
                    'file_path': file_path,
                    'objects': objects
                })
    return annotations

# Define paths to directories
train_dir = r"C:\Users\lenovo\OneDrive\Desktop\blade_damage_dataset\train"
val_dir = r"C:\Users\lenovo\OneDrive\Desktop\blade_damage_dataset\val"
test_dir = r"C:\Users\lenovo\OneDrive\Desktop\blade_damage_dataset\test"

annotation_dirs = [train_dir, val_dir, test_dir]
annotations = load_annotations(annotation_dirs)
print("Loaded annotations:", annotations)
