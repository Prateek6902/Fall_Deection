# **🧠 FSmart Fall Detection System**
An advanced fall detection web application built with Flask, YOLOv8, and a CNN classifier to identify human fall states (fall, not fall, sitting) from images.
The system allows users to upload images, performs detection and classification, and provides detailed reports with annotated images, confusion matrix, and performance metrics.

## ***📂 Project Structure***
```fall-detection/
│
├── app.py                # Flask backend application
├── models/               # Pre-trained YOLOv8 and CNN model files
│     ├── best.pt
│     └── best_cnn_fall_detection.pth
│
├── static/               # Static files (CSS, JS, images)
│     ├── css/
│     │     └── style.css
│     ├── js/
│     │     └── script.js
│     └── images/
│           └── results/  # Generated annotated results
│
├── templates/            # HTML templates
│     └── index.html
│
├── uploads/              # Uploaded files (images/videos)
│
└── requirements.txt      # Python dependencies

```
## ***🚀 Features***
YOLOv8 Object Detection: Detects persons in an image and identifies potential fall states.

CNN Posture Classification: Refines YOLO predictions for higher accuracy.

Detailed Reports:

Annotated image with bounding boxes and class labels.

Detection summary with confidence scores.

Confusion matrix visualization.

Performance metrics chart (Accuracy, Precision, Recall, F1 Score).

Responsive UI with intuitive upload and result display.

Support for Images (video/live stream coming soon).



## ***🛠️ Technologies Used***
Backend:
Python 3.x

Flask

OpenCV

PyTorch & TorchVision

Ultralytics YOLO

Matplotlib

PIL (Pillow)

NumPy

Frontend:
HTML5 / CSS3 (Responsive, custom styling)

JavaScript (Fetch API)

Font Awesome Icons

Google Fonts
## *** 📂 How It Works***
Upload Image

Select an image with people (standing, sitting, or falling) and click Process Image.

YOLOv8 Detection

Detects people and assigns an initial class label (fall, not_fall, sitting).

CNN Refinement

Crops detected person region and refines classification with a CNN model.

Report Generation

Annotated image saved in static/images/results/.

JSON response sent to frontend with detection data, confusion matrix, and metrics chart.

Frontend Rendering

Displays annotated image, detection summary, bounding box details, and probability analysis.

## *** 📊 Example Output***
Annotated Image:
Bounding boxes color-coded:

Red → Fall

Green → Not Fall

Blue → Sitting

Detection Summary:
YOLO Initial Detection + confidence

Final CNN Classification + confidence

Bounding box coordinates

CNN probability breakdown

Charts:
Confusion Matrix (sample data for demo)

Performance Metrics bar chart
![Minimalist Pancake Day Mood Photo Collage](https://github.com/user-attachments/assets/5a2e85fe-17e2-4531-beb1-91f7cca77b5b)


## ***📌 Use Cases***
Elderly fall detection in homes
Patient activity monitoring in hospitals
Smart surveillance in public places
Industrial worker safety compliance
