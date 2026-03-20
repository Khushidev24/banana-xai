# PES1UG23AM144_XAI_BANANA.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO

# Load Banana Image
url = "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
image = Image.open(BytesIO(response.content)).convert('RGB')

plt.imshow(image)
plt.axis('off')
plt.title("Input Image")
plt.show()

# Preprocess Image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0)

# Load Pretrained Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# ============================
# Task 3: Feature Map Analysis
# ============================
features = []

def hook_fn(module, input, output):
    features.append(output)

model.layer4.register_forward_hook(hook_fn)

output = model(input_tensor)

feature_map = features[0].squeeze(0)
avg_activation = feature_map.mean(dim=(1, 2))
max_channel = avg_activation.argmax()

print("Max channel:", max_channel.item())

selected_map = feature_map[max_channel].detach().numpy()

plt.imshow(selected_map, cmap='viridis')
plt.title("Most Activated Feature Map")
plt.colorbar()
plt.show()

# ============================
# Task 4: Grad-CAM
# ============================

def replace_relu(module):
    for name, layer in module.named_children():
        if isinstance(layer, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_relu(layer)

replace_relu(model)

gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer = model.layer4[-1]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

input_tensor.requires_grad_()

output = model(input_tensor)
pred_class = output.argmax()

model.zero_grad()
output[0, pred_class].backward()

grads = gradients[0]
acts = activations[0]

weights = grads.mean(dim=(2, 3), keepdim=True)

cam = (weights * acts).sum(dim=1).squeeze()
cam = torch.relu(cam)
cam = cam / cam.max()
cam = cam.detach().numpy()

cam = cv2.resize(cam, (224, 224))

img = input_tensor.detach().squeeze().permute(1, 2, 0).numpy()

plt.imshow(img)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.axis('off')
plt.title("Grad-CAM")
plt.show()
