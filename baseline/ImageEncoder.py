import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class BaseImageEncoder(nn.Module):
    def __init__(self, output_size=512):
        super(BaseImageEncoder, self).__init__()
        # self.input_size = input_size
        self.output_size = output_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # self.fc = nn.Linear(512, self.output_size).to(self.device)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 as expected by ResNet
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ResNet's requirements
        ])

    def forward(self, x): 
#         x = x.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            x = self.resnet(x.to(device))
        x =  x.view(x.size(0), -1)
        return x