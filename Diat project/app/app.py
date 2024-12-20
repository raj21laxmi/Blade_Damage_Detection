from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Define the new paths for uploads and results
UPLOAD_FOLDER = r'C:\Users\lenovo\OneDrive\Desktop\Diat project\app\static\uploads'
RESULT_FOLDER = r'C:\Users\lenovo\OneDrive\Desktop\Diat project\app\static\results'

# Update Flask app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO(r'C:\Users\lenovo\OneDrive\Desktop\Diat project\outputs\runs\detect\train16\weights\best.pt')

# Route for the main page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route to handle image upload
@app.route("/upload", methods=["POST"])
def upload_image():
    if request.method == "POST":
        # Get the uploaded image
        file = request.files.get("image")
        if not file:
            print("No file received in the request.")
            return "No file uploaded."

        # Debug: Show received file details
        print(f"Received file: {file.filename}")

        # Construct file paths
        filename = file.filename
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)

        # Debug: Check paths
        print(f"Upload path: {upload_path}")
        print(f"Result path: {result_path}")

        # Save the uploaded file
        try:
            file.save(upload_path)
            print(f"Uploaded file saved successfully at: {upload_path}")
        except Exception as e:
            print(f"Error saving uploaded file: {e}")
            return "Error saving uploaded file."

        # Run YOLO detection
        try:
            results = model(upload_path)
            annotated_image = results[0].plot()  # Annotate the image with detections

            # Save the result image
            cv2.imwrite(result_path, annotated_image)
            print(f"Result image saved successfully at: {result_path}")
        except Exception as e:
            print(f"Error during YOLO detection or saving result: {e}")
            return "Error processing the uploaded image."

        # Redirect to result page
        return redirect(url_for("result", filename=filename))

# Route to display the results
@app.route("/result/<filename>", methods=["GET"])
def result(filename):
    # Correctly reference the static file paths
    uploaded_image_url = f"/static/uploads/{filename}"
    result_image_url = f"/static/results/{filename}"
    return render_template("result.html", uploaded_image_url=uploaded_image_url, result_image_url=result_image_url)

if __name__ == "__main__":
    app.run(debug=True)
