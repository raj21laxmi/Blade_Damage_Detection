# Blade Damage Detection Project

This project is a YOLOv8-based blade damage detection system designed to detect cracks and damage on blade surfaces using deep learning. The project includes a user-friendly frontend, a well-structured backend, and model training and inference capabilities.

## Features
- Detects blade damages (cracks) in uploaded images.
- Displays annotated results with bounding boxes.

Project Directory Structure :

DIAT_Project/
├── app/                     # Web application files
│   ├── app.py               # Flask app entry point
│   ├── templates/           # HTML templates for the web app
│   │   ├── index.html       # Main upload page
│   │   └── result.html      # Detection results display
│   └── static/              # Static files (CSS, JS, images)
│       ├── css/             # CSS files for styling
│       ├── uploads/         # Uploaded images
│       └── results/         # YOLO inference results
├── config/                  # Configuration files
│   ├── dataset.yaml         # Dataset configuration for YOLO
│   ├── hyp.yaml             # Hyperparameter settings for YOLO
│   └── blade_damage.yaml    # Class definitions (if applicable)
├── dataset/                 # Dataset for training and testing
│   ├── train/               # Training images and labels
│   │   ├── images/
│   │   └── labels/
│   ├── val/                 # Validation images and labels
│   │   ├── images/
│   │   └── labels/
│   └── test/                # Test images and labels
│       ├── images/
│       └── labels/
├── models/                  # YOLO model files
│   └── best.pt              # Trained YOLOv8 weights
├── outputs/                 # Outputs and logs
│   ├── detection_results/   # Final detection results
│   └── runs/                # YOLO training and inference runs
├── src/                     # Source files for data preprocessing and utilities
│   ├── detect_single_image.py  # Script for single image detection
│   ├── train_yolo.py            # YOLO training script
│   ├──....                        # Other utility scripts
├── yolo_env/                # YOLO environment setup files
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies


## How to Run
1. Install dependencies using `pip install -r requirements.txt`.
2. Run the Flask app: `python app.py`.
3. Visit `http://127.0.0.1:5000` in your browser.

## Requirements
See `requirements.txt` for a full list of dependencies.

## Results
![Example](app/static/results/example.jpg)
