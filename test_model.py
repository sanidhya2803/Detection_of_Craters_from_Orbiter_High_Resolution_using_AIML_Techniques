import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import os

# Function to load the trained model
def load_model(model_path, num_classes):
    # Load the model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Load your trained model
model_path = 'faster_rcnn_model.pth'  # Change to your model file path
num_classes = 2  # Update this based on your training (1 for object + 1 for background)
model = load_model(model_path, num_classes)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Example: Load your test image
test_image_path = 'test/images/lunar_crater_test_7.jpg'  # Change to your test image path
test_image = load_image(test_image_path).to(device)

# Function to predict and display results
def predict_and_display(image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Display predictions (boxes, labels, and scores)
    for element in predictions[0]['boxes']:
        print('Bounding Box:', element.cpu().numpy())  # Print the bounding box coordinates

    for element in predictions[0]['labels']:
        print('Label:', element.cpu().numpy())  # Print the predicted label

    for element in predictions[0]['scores']:
        print('Score:', element.cpu().numpy())  # Print the score of the prediction

# Run prediction on the test image
predict_and_display(test_image)

import matplotlib.pyplot as plt

def visualize_predictions(image_tensor, predictions):
    image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    plt.imshow(image)
    ax = plt.gca()

    # Draw bounding boxes
    for box, label in zip(predictions[0]['boxes'], predictions[0]['labels']):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     fill=False, color='red', linewidth=2))
        ax.text(x_min, y_min, f'Label: {label.item()}', color='red', fontsize=12)

    plt.axis('off')
    plt.show()

# Update predict_and_display function to visualize predictions
def predict_and_display(image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)

    visualize_predictions(image_tensor, predictions)

# Run prediction on the test image
predict_and_display(test_image)

