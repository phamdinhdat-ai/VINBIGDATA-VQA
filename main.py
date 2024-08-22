import os 
import pandas as pd 
import glob 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
# from torchvision import transforms


from dataloader.VQADataset import VQADataset
from models.model import VQAModel
from utils.utils import dataloader_json

root = './vizwiz'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = VQAModel().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)


# transforms = transforms.Compose([transforms.Resize((224, 224)),
#                                  #transforms.CenterCrop(224),
#                                  transforms.ToTensor(),
#                                  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                # ])
# train_dataset = VQADataset(root=root, mode ="train", transform=None)
# val_dataset = VQADataset(root=root, mode="val", transform=None)
# test_dataset = VQADataset(root=root, mode="val", transform=None)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters.')


