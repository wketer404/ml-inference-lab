import time
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
from torchvision.models import ResNet50_Weights


device = torch.device("cuda")

weights = ResNet50_Weights.DEFAULT
preprocess =weights.transforms()

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
    "models/resnet50_fp32.onnx",
    providers=["CUDAExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

images, _ = next(iter(loader))
images = images.to(device, non_blocking=True).contiguous()

output = torch.empty(
    (images.shape[0], 1000),
    dtype=torch.float32,
    device=device,
).contiguous()

io_binding = session.io_binding()

io_binding.bind_input(
    name=input_name,
    device_type="cuda",
    device_id=0,
    element_type=np.float32,
    shape=tuple(images.shape),
    buffer_ptr=images.data_ptr(),
)

io_binding.bind_output(
    name=output_name,
    device_type="cuda",
    device_id=0,
    element_type=np.float32,
    shape=tuple(output.shape),
    buffer_ptr=output.data_ptr(),
)

session.run_with_iobinding(io_binding)

torch.cuda.synchronize()

num_iterations = 100

torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(num_iterations):
    session.run_with_iobinding(io_binding)

torch.cuda.synchronize()
end = time.perf_counter()

total_time = end-start
latency = total_time / num_iterations

batch_size = images.shape[0]
throughput = batch_size * num_iterations / total_time

print(f"Average batch latency: {latency * 1000:.2f} ms")
print(f"Throughput: {throughput:.2f} images/sec")