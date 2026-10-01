import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
from torchvision.models import ResNet50_Weights, resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

dataset = Imagenette(
    root="data",
    split="val",
    size="160px",
    download=False,
    transform=preprocess,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

imagenette_to_imagenet = torch.tensor([
    0,    # tench
    217,  # English springer
    482,  # cassette player
    491,  # chain saw
    497,  # church
    566,  # French horn
    569,  # garbage truck
    571,  # gas pump
    574,  # golf ball
    701,  # parachute
])

model = resnet50(weights=weights)
model.to(device)
model.eval()

correct = 0
total = 0

with torch.inference_mode():
    for images, labels in loader:

        true_labels = imagenette_to_imagenet[labels]
        
        images = images.to(device, non_blocking=True)
        true_labels = true_labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == true_labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100

print(f"Correct: {correct}/{total}")
print(f"Top-1 accuracy: {accuracy:.2f}%")