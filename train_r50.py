import os
import torch
import torchvision.models as models
from transformers import DetrConfig, DetrForObjectDetection
from torchvision.models import VGG16_Weights

# Path configurations
HOME = os.getcwd()
MODEL_PATH = os.path.join(HOME, 'detr_api-1')
ANNOTATION_FILE_NAME = "_annotations.coco.json"

# Model configurations
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8

# Training configurations
MAX_EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 3

# API Keys
ROBOFLOW_API_KEY = "umCDBYYeGbFwUd9x2KRY"

# Load VGG-16 as backbone
class VGGBackbone(torch.nn.Module):
    def __init__(self):
        super(VGGBackbone, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = vgg.features  # Use only the convolutional layers
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to handle varying input sizes

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

# Replace DETR's ResNet backbone with VGG
class DETRWithVGG(torch.nn.Module):
    def __init__(self, num_classes):
        super(DETRWithVGG, self).__init__()
        self.backbone = VGGBackbone()
        self.config = DetrConfig(num_classes=num_classes)
        self.detr = DetrForObjectDetection(self.config)

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.detr(features)
        return outputs

# Initialize DETR with VGG backbone
num_classes = 91  # COCO dataset has 80 classes + 1 background class
model = DETRWithVGG(num_classes=num_classes).to(DEVICE)

# Print model summary
print(model)