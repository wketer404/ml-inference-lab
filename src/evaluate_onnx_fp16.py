import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
from torchvision.models import ResNet50_Weights


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

session = ort.InferenceSession(
    "models/resnet50_fp16.onnx",
    providers=["CUDAExecutionProvider"],
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

correct = 0
total = 0

with torch.inference_mode():
    for images, labels in loader:
        inputs = images.numpy().astype(np.float16)

        outputs = session.run(
            ["logits"],
            {"images": inputs},
        )[0]

        predictions = np.argmax(outputs, axis=1)

        true_labels = np.array(
            [imagenette_to_imagenet[label.item()] for label in labels]
        )

        correct += np.sum(predictions == true_labels)
        total += labels.size(0)

print(f"Correct: {correct}/{total}")
print(f"Top-1 accuracy: {100 * correct / total:.2f}%")