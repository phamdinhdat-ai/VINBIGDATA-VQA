import torch 
import os 
import json
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from utils.utils import find_most_common_answer, select_most_common_answers , segment_text



root = "./vizvwiz"

class VQADataset(Dataset):
    def __init__(self, root, mode = 'train', transform = None ):
        self.mode = mode.lower()
        self.root = root 
        self.transform = transform
        self.data = self.load_json()
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(" Mode should be train , val or test")
        
        
    def load_json(self):
        with open(f"{self.root}/Annotations/Annotations/{self.mode}.json", 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        
        if self.mode == 'test':
            return df 
        else:
            return select_most_common_answers(df)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, question, list_answers , answer_type, answerable, answer = self.data[index]
        base_path = f"{self.root}/{self.mode}/{self.mode}/{img_path}"
        img = Image.open(base_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, question, list_answers, answer_type, answerable, answer
        
