import os
import torch
from flask import Flask, request, render_template, send_from_directory
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib to avoid Tkinter issues
import matplotlib.pyplot as plt
import numpy as np
import io

app = Flask(__name__)

# Load the trained Faster R-CNN model
def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load the model
model_path = 'faster_rcnn_model.pth'  # Path to the trained model
num_classes = 2  # 1 for object + 1 for background
model = load_model(model_path, num_classes)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Predict and draw bounding boxes with scores
def predict_and_display(image_tensor, score_threshold=0.7):
    with torch.no_grad():
        predictions = model(image_tensor)

    image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    ax = plt.gca()

    # Iterate over predictions and filter based on score_threshold
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > score_threshold:  # Apply confidence threshold
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                       fill=False, color='red', linewidth=2))
            ax.text(x_min, y_min, f'Score: {score:.2f}', color='red', fontsize=12)

    plt.axis('off')

    # Save the image with bounding boxes
    result_image_path = 'static/predictions/prediction_result.jpg'
    plt.savefig(result_image_path)
    plt.close()

    return result_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    # Save the uploaded file
    upload_folder = 'static/uploaded_images'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Load and predict
    image_tensor = load_image(file_path).to(device)
    result_image_path = predict_and_display(image_tensor)

    return render_template('result.html', uploaded_image=file_path, result_image=result_image_path)

# Serve the uploaded and result images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploaded_images', filename)

@app.route('/predictions/<filename>')
def result_file(filename):
    return send_from_directory('static/predictions', filename)

if __name__ == '__main__':
    app.run(debug=True)
