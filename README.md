# Banana Classification using Explainable AI

## Student Details
- Name: Khushi Dev  
- SRN: PES1UG23AM144  

## Project Description
This project demonstrates how a pretrained Convolutional Neural Network (ResNet18) identifies and classifies a banana image using Explainable AI techniques.

## Tasks

### Task 3: Feature Map Analysis
- Extracted feature maps from layer4 of ResNet18
- Computed average activation for each channel
- Identified the most activated feature map
- Visualized internal representations of the model

### Task 4: Grad-CAM Visualization
- Applied Grad-CAM on the last convolutional layer
- Generated heatmap using gradients and activations
- Highlighted important regions contributing to prediction

## Model Used
- ResNet18 (Pretrained)

## Output
- Feature map showing strongest activation
- Grad-CAM heatmap focusing on banana region

## Conclusion
- Model successfully identifies banana features like shape and texture
- Grad-CAM confirms model focuses on relevant regions
- Background influence is minimal

## How to Run
pip install torch torchvision matplotlib opencv-python pillow requests  
python PES1UG23AM144_XAI_BANANA.py
