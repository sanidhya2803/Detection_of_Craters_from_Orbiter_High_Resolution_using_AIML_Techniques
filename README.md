# 🚀 Detection of Craters from Orbiter High Resolution using Faster R-CNN

This project aims to automate the detection of lunar surface craters using a deep learning-based object detection model, **Faster R-CNN**, with **ResNet-50** as the backbone. Built entirely from scratch, the model is trained on high-resolution lunar images captured by the **Orbiter High Resolution Camera (OHRC)** and annotated in **COCO format**.

## 📌 Project Objective

- To develop an AI-powered system that can **detect and localize lunar craters** from satellite imagery.
- To assist in **lunar terrain analysis**, **mission planning**, and **autonomous navigation** by providing accurate surface information.
- To demonstrate the use of **custom-built Faster R-CNN with ResNet-50**, trained from scratch without pre-trained weights.

## 🧠 Key Features

- ✅ Custom implementation of **Faster R-CNN**.
- ✅ Backbone: **ResNet-50** (built from scratch).
- ✅ Dataset in **COCO format** containing OHRC images of the lunar surface.
- ✅ Evaluation based on **Precision**, **Recall**, **mAP** (Mean Average Precision).
- ✅ Visualization of detection results.
- ✅ Option for real-time inference via web interface or Gradio app.

## 📊 Dataset

- **Source**: High-resolution images from Orbiter High Resolution Camera (OHRC)
- **Format**: COCO (`.json`) with `images`, `annotations`, and `categories`
- **Classes**: 
  - `crater` (label ID: 1)

## 🏗️ Model Architecture

- **Backbone**: ResNet-50 (custom)
- **Region Proposal Network (RPN)**: Generates object proposals
- **RoI Align + Classifier Head**: Refines proposals and classifies craters

## ⚙️ Installation & Setup

1. **Clone the Repository**
   bash
   git clone https://github.com/yourusername/crater-detection-fasterrcnn.git
   cd crater-detection-fasterrcnn

2. **Create Virtual Environment**
    bash
    Copy code
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows

3. **Install Dependencies**
    bash
    Copy code
    pip install -r requirements.txt

**Training the Model**
  bash
  Copy code
  python train.py --config configs/train_config.yaml
  Ensure the dataset paths and hyperparameters are correctly set in the config file.

📈 **Evaluation**
  bash
  Copy code
  python evaluate.py --weights path/to/model_weights.pth
  Metrics such as Precision, Recall, and mAP will be displayed.

🌐** Run the Web App (Flask)**
  bash
  Copy code
  cd app
  python app.py
