import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import roi_pool
from PIL import Image
import json
import os

# Backbone Network (ResNet-50) 
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super(ResNet50Backbone, self).__init__()
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_planes, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# RPN (Region Proposal Network)
class RPN(nn.Module):
    def __init__(self, in_channels=512, num_anchors=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(512, num_anchors * 2, kernel_size=1, stride=1)
        self.reg_layer = nn.Conv2d(512, num_anchors * 4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        cls_out = self.cls_layer(x)
        reg_out = self.reg_layer(x)
        return cls_out, reg_out

# Faster R-CNN Model
class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        self.backbone = ResNet50Backbone()
        self.rpn = RPN(in_channels=512)
        self.roi_pool = roi_pool  # Use torchvision's roi_pool function
        self.classifier = nn.Linear(512 * 7 * 7, num_classes)
        self.bbox_regressor = nn.Linear(512 * 7 * 7, num_classes * 4)

    def forward(self, images, rois):
        feature_map = self.backbone(images)
        cls_logits, bbox_preds = self.rpn(feature_map)
        pooled_rois = self.roi_pool(feature_map, rois, output_size=(7, 7))
        pooled_rois = pooled_rois.view(pooled_rois.size(0), -1)
        class_scores = self.classifier(pooled_rois)
        bbox_deltas = self.bbox_regressor(pooled_rois)
        return class_scores, bbox_deltas

# Example Data Loading (COCO Format)
class CocoCraterDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_path = os.path.join(self.img_dir, image_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        img_id = image_info['id']
        annos = [anno for anno in self.annotations if anno['image_id'] == img_id]
        valid_boxes = []
        for anno in annos:
            x_min, y_min, width, height = anno['bbox']
            if width > 0 and height > 0:
                valid_boxes.append([x_min, y_min, x_min + width, y_min + height])
        image = torchvision.transforms.functional.to_tensor(image)
        return image, valid_boxes

# Training Loop with Model Saving
def train(model, dataloader, device, optimizer, num_epochs=5, save_path="faster_rcnn_model_epoch_{}.pth"):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            rois = [torch.tensor(target, dtype=torch.float32).to(device) for target in targets]
            optimizer.zero_grad()
            class_scores, bbox_deltas = model(images, rois)
            # Implement losses (classification and regression)
            # Losses here are just placeholders and need to be defined
            loss = torch.tensor(0.0, requires_grad=True).to(device)  # Placeholder
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")
        
        # Save model checkpoint at the end of each epoch
        torch.save(model.state_dict(), save_path.format(epoch + 1))
        print(f"Model saved at epoch {epoch+1}")

# Instantiate Model and Training Components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Background + 1 crater class
model = FasterRCNN(num_classes).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Load Data
train_image_dir = 'train/images'
train_ann_file = 'train/_annotations.coco.json'
train_dataset = CocoCraterDataset(train_image_dir, train_ann_file)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Start Training
train(model, train_loader, device, optimizer, num_epochs=10)
