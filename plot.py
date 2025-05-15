import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

# Example image path
image_path = 'test\images\lunar_crater_test_8.jpg'  # Update this to your actual image path

# Example prediction (as given in your output)
output = {
    'boxes': torch.tensor([[510.3414,  71.1308, 565.6689, 126.3074],
        [183.7585,  84.2377, 551.4667, 437.6439],
        [ 45.8143,  49.3488, 220.5564, 234.5972],
        [ 85.2710, 151.3005, 162.0986, 222.0460],
        [335.7807, 383.2991, 433.8788, 432.5386],
        [ 75.5476,  52.7163, 177.8866, 206.5818],
        [574.7172, 556.6912, 627.4036, 609.5396],
        [ 93.9019,  80.3350, 238.5839, 182.6772],
        [ 26.2713,  20.5467, 223.6445, 364.3551],
        [236.2588,  92.6804, 557.8000, 300.7721],
        [ 95.8935, 158.6719, 139.1921, 201.8085],
        [151.6628,  92.5848, 207.9902, 184.2140],
        [ 96.4042,  74.3385, 199.9452, 251.3462],
        [ 79.5777, 144.3345, 187.7897, 250.3105],
        [513.9860,  62.0488, 588.2916, 157.0762]]), 'labels': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'scores': torch.tensor([0.9394, 0.9017, 0.6942, 0.4524, 0.3639, 0.2770, 0.1941, 0.1343, 0.1069,
        0.1047, 0.0742, 0.0726, 0.0701, 0.0564, 0.0563])
}

# Set the threshold for confidence scores to filter boxes
threshold = 0.5

# Load the image
image = Image.open(image_path)

# Create a Matplotlib figure
fig, ax = plt.subplots(1)
ax.imshow(image)

# Loop through predictions and draw boxes with confidence scores above the threshold
for i, box in enumerate(output['boxes']):
    score = output['scores'][i].item()
    if score > threshold:
        # Get the bounding box coordinates
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        
        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add confidence score as text
        ax.text(x_min, y_min - 5, f'{score:.2f}', color='red', fontsize=12, weight='bold')

# Display the image with bounding boxes
plt.show()