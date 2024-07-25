"""Gets the current class names of the model"""

import torch
from pathlib import Path

def get_class_names(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    params_bd = checkpoint['params']
    class_names = params_bd["class_names"]
    return class_names

if __name__ == "__main__":
    model_path = Path(r"C:\wagon\code\biosonic_local\batdetect2\batdetect2\models\Net2DFast_UK_same.pth.tar")
    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
    else:
        class_names = get_class_names(model_path)
        print("Class names:", class_names)