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

## Current Progress

- [x] Set up the Python/PyTorch environment
- [x] Load pretrained ResNet-50
- [x] Run single-image inference
- [x] Display top-5 ImageNet predictions
- [x] Accept image paths through the command line
- [x] Evaluate on a dataset
- [x] Establish dataset-level Top-1 accuracy baseline
- [x] Complete PyTorch FP32 GPU benchmarking
- [x] Complete PyTorch FP16 benchmarking
- [x] Export to ONNX
- [x] Benchmark ONNX Runtime
- [ ] Add TensorRT
- [ ] Add INT8 quantization
- [ ] Profile GPU performance

## Benchmark Environment

### Hardware

- GPU: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM
- CPU: 2× Intel Xeon E5-2630 v4 @ 2.20 GHz
- CPU cores: 20 physical cores / 40 threads
- RAM: 16 GB allocated to the container

### Software

- OS/Environment: UCSD DSMLP
- NVIDIA Driver: 535.161.08
- CUDA: 12.2
- PyTorch: 2.2.1+cu121
- Python: 3.11

## Current Results

| Configuration | Dataset | Images | Accuracy | Batch Size | Latency / Batch | Throughput | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | Imagenette validation set | 3,925 | 80.31% | 32 | 49.97 ms | 640.44 img/s | 442.79 MB |
| PyTorch FP16 | Imagenette validation set | 3,925 | 80.38% | 32 | 53.74 ms | 595.49 img/s | 267.74 MB |

> Latency and throughput were measured using 100 repeated inference iterations on a fixed batch after GPU warm-up. Peak VRAM refers to PyTorch-allocated GPU memory.

## Current Pipelines

Single-image inference:

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

Dataset evaluation:

```text
Imagenette validation set
  ↓
Preprocessing and DataLoader
  ↓
Batches of [32, 3, 224, 224]
  ↓
ResNet-50
  ↓
Top-1 predictions and accuracy
```

GPU benchmarking:

```text
Fixed input batch → GPU → ResNet-50 → repeated inference ×100 → latency / throughput / peak VRAM
```

## Project Structure

```text
ml-inference-lab/
├── src/
│   ├── inference.py
│   ├── evaluate.py
│   └── benchmark.py
├── images/
├── data/              # gitignored
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

Run `python src/evaluate.py` to download the Imagenette dataset into `data/` and evaluate the validation set. The `data/` directory is gitignored.

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

## Benchmark Configurations

Configurations are compared using a common benchmark. Latency is per batch of 32 images; unmeasured values remain TBD.

| Runtime | Precision | Accuracy | Batch Size | Latency / Batch | Throughput | GPU Memory | Artifact Size |
|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch | FP32 | 80.31% | 32 | 49.97 ms | 640.44 img/s | 442.79 MiB | TBD |
| PyTorch | FP16 | 80.38% | 32 | 53.74 ms | 595.49 img/s | 267.74 MiB | TBD |
| ONNX Runtime | FP32 | 80.31% | 32 | 43.49 ms | 735.82 img/s | ~712 MiB | 98 MiB |
| ONNX Runtime | FP16 | 80.36% | 32 | 44.54 ms | 718.41 img/s | ~748 MiB | 49 MiB |
| TensorRT | FP32 | TBD | 32 | TBD | TBD | TBD | TBD |
| TensorRT | FP16 | TBD | 32 | TBD | TBD | TBD | TBD |
| TensorRT | INT8 | TBD | 32 | TBD | TBD | TBD | TBD |

> Memory measurements are not directly comparable across runtimes. PyTorch values represent peak PyTorch-allocated GPU memory measured with `torch.cuda.max_memory_allocated()`, while the ONNX Runtime value represents the increase in total GPU memory reported by `nvidia-smi` from a 0 MiB baseline.

## Technologies

Current:

- Python
- PyTorch
- TorchVision
- Pillow
- CUDA

Planned:

- ONNX
- ONNX Runtime
- NVIDIA TensorRT
- GPU profiling tools

## Possible Future Work

- Model pruning
- Knowledge distillation
- Comparison with other architectures
- Custom CUDA kernels

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

## Status

Work in progress.
