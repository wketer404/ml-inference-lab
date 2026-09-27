from PIL import Image

import torch
from torchvision.models import ResNet50_Weights, resnet50

# Load pretrained weights
weights = ResNet50_Weights.DEFAULT

# Load pretrained ResNet-50
model = resnet50(weights=weights)
model.eval()

# Get the preprocessing expected by these weights
preprocess = weights.transforms()

# Load an image
image = Image.open("images/test.jpg").convert("RGB")

# Convert image -> tensor and add btach dimension
input_tensor = preprocess(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Convert output scores to probabilities
probabilities = torch.softmax(output[0], dim=0)