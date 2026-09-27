# ML Inference Lab

A hands-on machine learning systems project focused on optimizing neural network inference performance.

The goal is to take a pretrained vision model, establish a reproducible performance baseline, apply different inference optimizations, and measure the tradeoffs between speed, memory usage, and model accuracy.

## Project Goal

We want to answer a simple question:

> How much can we improve neural network inference performance without significantly hurting accuracy?

The project will compare different inference configurations across metrics such as:

- Accuracy
- Latency
- Throughput
- GPU memory usage
- Model size

## Roadmap

The project will progress through the following stages:

1. Run pretrained ResNet-50 inference on individual images
2. Evaluate the model on a dataset
3. Establish a PyTorch FP32 baseline
4. Benchmark GPU latency and throughput
5. Test FP16 inference
6. Export the model to ONNX
7. Benchmark ONNX Runtime
8. Optimize inference with TensorRT
9. Test INT8 quantization
10. Profile GPU performance and bottlenecks
11. Compare all configurations
12. Explore advanced optimizations such as pruning, distillation, or custom CUDA kernels

## Current Progress

- [x] Set up the Python/PyTorch environment
- [x] Load pretrained ResNet-50
- [x] Run single-image inference
- [x] Display top-5 ImageNet predictions
- [x] Accept image paths through the command line
- [ ] Evaluate on a dataset
- [ ] Build baseline benchmark
- [ ] Add FP16 inference
- [ ] Export to ONNX
- [ ] Add TensorRT
- [ ] Add INT8 quantization
- [ ] Profile GPU performance

## Current Pipeline

```text
Image
  ↓
Preprocessing
  ↓
Tensor [1, 3, 224, 224]
  ↓
ResNet-50
  ↓
1000 ImageNet class scores
  ↓
Softmax
  ↓
Top-5 predictions
```

## Project Structure

```text
ml-inference-lab/
├── src/
│   └── inference.py
├── images/
├── README.md
└── .gitignore
```

## Setup

Clone the repository:

```bash
git clone https://github.com/wketer404/ml-inference-lab.git
cd ml-inference-lab
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision pillow
```

## Running Inference

Place an image inside the `images/` directory.

Then run:

```bash
python src/inference.py images/example.jpg
```

Example output:

```text
banana: 62.81%
spaghetti squash: 0.22%
orange: 0.14%
grocery store: 0.10%
coil: 0.08%
```

The model currently uses a pretrained ResNet-50 trained on ImageNet-1K and returns predictions from its 1,000 supported classes.

## Planned Benchmarking

Each optimization will eventually be compared using a common benchmark:

| Configuration | Accuracy | Latency | Throughput | VRAM | Model Size |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | TBD | TBD | TBD | TBD | TBD |
| PyTorch FP16 | TBD | TBD | TBD | TBD | TBD |
| ONNX Runtime | TBD | TBD | TBD | TBD | TBD |
| TensorRT FP16 | TBD | TBD | TBD | TBD | TBD |
| TensorRT INT8 | TBD | TBD | TBD | TBD | TBD |

## Technologies

- Python
- PyTorch
- TorchVision
- ONNX
- ONNX Runtime
- NVIDIA TensorRT
- CUDA
- GPU profiling tools

## Learning Goals

This project is intended to build experience with:

- Neural network inference
- PyTorch
- Computer vision models
- GPU benchmarking
- Numerical precision
- Quantization
- Model deployment
- CUDA/GPU performance
- Profiling and bottleneck analysis
- Reproducible ML experimentation

## Contributors

- Keter Wu
- Additional contributor(s)

## Status

Work in progress.