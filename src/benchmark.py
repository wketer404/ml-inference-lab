import time
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

model = resnet50(weights=weights)
model.to(device)
model.eval()

with torch.inference_mode():
    # Warm up the GPU
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        model(images)
        break

    images, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)

    num_iterations = 100

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(num_iterations):
            model(images)

    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    latency = total_time / num_iterations
    batch_size = images.shape[0]
    throughput = batch_size * num_iterations / total_time

    print(f"Average batch latency: {latency * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")