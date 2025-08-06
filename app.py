import os
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Configuration
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'STATIC_FOLDER': 'static',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'mp4'},
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'MODEL_DIR': 'models'
})

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'images/results'), exist_ok=True)

class FallDetectionCNN(torch.nn.Module):
    """CNN model for fall detection classification"""
    def __init__(self, num_classes=3):
        super(FallDetectionCNN, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.model.fc.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def load_models():
    """Load and initialize both models with error handling"""
    models = {}
    try:
        # Load YOLOv8 model
        yolo_model_path = os.path.join(app.config['MODEL_DIR'], 'best.pt')
        models['yolo'] = YOLO(yolo_model_path)
        
        # Verify model classes
        print(f"YOLO Model Classes: {models['yolo'].names}")
        
        # Load CNN model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn_model_path = os.path.join(app.config['MODEL_DIR'], 'best_cnn_fall_detection.pth')
        
        models['cnn'] = FallDetectionCNN(num_classes=3).to(device)
        models['cnn'].load_state_dict(torch.load(cnn_model_path, map_location=device))
        models['cnn'].eval()
        
        # Define CNN preprocessing
        models['cnn_transform'] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Models loaded successfully")
        return models
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise

models = load_models()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_fall_states(image_path):
    """Detect fall states using YOLOv8 with color-coded bounding boxes based on final classification"""
    results = models['yolo'](image_path)
    detections = []
    
    # Create a copy of the original image for annotation
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError("Could not read image")
    
    annotated_img = original_img.copy()
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Define colors for each class (BGR format)
    class_colors = {
        'fall': (0, 0, 255),      # Red
        'not_fall': (0, 255, 0),   # Green
        'sitting': (255, 0, 0)     # Blue
    }
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            yolo_class = models['yolo'].names[class_id]
            confidence = float(box.conf[0])
            box_coords = box.xyxy[0].tolist()
            
            # Get the image crop for CNN classification
            x1, y1, x2, y2 = map(int, box_coords)
            person_img = rgb_img[y1:y2, x1:x2]
            
            # Default to YOLO classification
            final_class = yolo_class
            final_confidence = confidence
            
            # Use CNN for refinement if we have a valid crop
            if person_img.size > 0:
                cnn_result = classify_with_cnn(person_img)
                final_class = cnn_result['class']
                final_confidence = cnn_result['confidence'] / 100  # Convert from percentage
            
            # Store detection information
            detection = {
                'yolo_class': yolo_class,
                'yolo_confidence': confidence,
                'final_class': final_class,
                'final_confidence': final_confidence,
                'box': box_coords,
                'cnn_probs': {
                    'fall': cnn_result.get('prob_0', 0),
                    'not_fall': cnn_result.get('prob_1', 0),
                    'sitting': cnn_result.get('prob_2', 0)
                } if person_img.size > 0 else None
            }
            detections.append(detection)
            
            # Draw bounding box with FINAL classification color
            color = class_colors.get(final_class, (255, 255, 255))  # Default to white if class not found
            
            # Draw rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Put class label with confidence
            label = f"{final_class.replace('_', ' ')} {final_confidence:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return detections, annotated_img

def classify_with_cnn(image_array):
    """Classify posture using CNN model with confidence scores"""
    try:
        img = Image.fromarray(image_array)
        img = models['cnn_transform'](img).unsqueeze(0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = img.to(device)
        
        with torch.no_grad():
            outputs = models['cnn'](img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            # Get all class probabilities
            all_probs = {f'prob_{i}': round(probs[0][i].item() * 100, 2) for i in range(3)}
        
        classes = ['fall', 'not_fall', 'sitting']
        return {
            'class': classes[pred.item()],
            'confidence': round(conf.item() * 100, 2),
            **all_probs  # Include all class probabilities
        }
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return {'class': 'error', 'confidence': 0.0}

def generate_confusion_matrix():
    """Generate sample confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    cm = np.array([[85, 5, 10], [3, 90, 7], [8, 2, 90]])  # Sample data
    classes = ['Fall', 'Not Fall', 'Sitting']
    
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', pad=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label', labelpad=10)
    plt.xlabel('Predicted Label', labelpad=10)
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_metrics_chart():
    """Generate sample metrics chart"""
    metrics = {
        'Accuracy': 94.5,
        'Precision': 93.2,
        'Recall': 92.8,
        'F1 Score': 93.0
    }
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#4285F4', '#34A853', '#EA4335', '#FBBC05'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval}%",
                ha='center', va='bottom', fontsize=10)
    
    plt.ylim(0, 100)
    plt.title('Model Performance Metrics', pad=20)
    plt.ylabel('Score (%)')
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Process image
                detections, annotated_img = detect_fall_states(file_path)
                
                # Use CNN for additional classification
                original_img = cv2.imread(file_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                for det in detections:
                    x1, y1, x2, y2 = map(int, det['box'])
                    person_img = original_img[y1:y2, x1:x2]
                    if person_img.size > 0:  # Only classify if we have a valid crop
                        cnn_result = classify_with_cnn(person_img)
                        det.update(cnn_result)  # Add all CNN results to the detection
                
                # Save annotated image
                result_img_path = os.path.join(app.config['STATIC_FOLDER'], 'images/results', filename)
                cv2.imwrite(result_img_path, annotated_img)
                
                return jsonify({
                    'type': 'image',
                    'status': 'success',
                    'result_img': f"static/images/results/{filename}",
                    'detections': detections,
                    'confusion_matrix': generate_confusion_matrix(),
                    'metrics_chart': generate_metrics_chart()
                })
            
            elif filename.lower().endswith('.mp4'):
                return jsonify({
                    'type': 'video',
                    'message': 'Video processing will be implemented soon',
                    'filename': filename
                })
        
        except Exception as e:
            return jsonify({
                'error': f'Processing failed: {str(e)}'
            }), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/live')
def live_feed():
    return jsonify({
        'message': 'Live feed processing will be implemented soon'
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)