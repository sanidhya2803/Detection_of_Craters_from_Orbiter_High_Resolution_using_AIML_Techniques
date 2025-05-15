import torch
import torchvision
from torchvision import models, transforms
import json
import os
from sklearn.metrics import f1_score
from PIL import Image

# Function to load COCO annotations
def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

# Function to load images
def load_images(image_dir):
    images = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(image_dir, img_file))
    return images

# Load the model
num_classes = 91  # Update this based on your dataset
model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT', num_classes=num_classes)
model.eval()

# Load annotations and images
annotation_file = 'test/_annotations.coco.json'  # Update with your annotation file path
image_dir = 'test/images'  # Update with your image directory path
annotations = load_coco_annotations(annotation_file)
images = load_images(image_dir)

# Prepare to collect true and predicted labels
true_labels = []
predicted_labels = []

# Load the model and process each image
for img_name in images:
    image = Image.open(img_name)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)

    # Process outputs
    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    labels = outputs[0]['labels']

    # Filter out low-confidence predictions (threshold = 0.5)
    high_confidence_indices = scores > 0.75
    boxes = boxes[high_confidence_indices]
    labels = labels[high_confidence_indices]

    # Convert to numpy for compatibility with sklearn
    predicted_labels.extend(labels.cpu().numpy())

    # Extract true labels for the current image
    img_id = annotations['images'][0]['id']  # Modify as needed to get correct image_id
    target = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]

    if target:
        true_labels.extend([ann['category_id'] for ann in target])

# Debugging output
print(f'True labels: {true_labels}')
print(f'Predicted labels: {predicted_labels}')
print(f'Number of true labels: {len(true_labels)}')
print(f'Number of predicted labels: {len(predicted_labels)}')

# Handle cases with no predictions
if true_labels and predicted_labels:
    # Ensure that the lists have consistent lengths for evaluation
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
else:
    f1 = 0.0

# Output results
print(f'F1 Score: {f1}')
