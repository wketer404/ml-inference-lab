import torch
from torchvision.models import ResNet50_Weights, resnet50


weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "models/resnet50_fp32.onnx",
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes={
        "images": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
)

print("Exported models/resnet50_fp32.onnx")