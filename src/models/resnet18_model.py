import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

def build_resnet18(num_classes=2, pretrained=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
