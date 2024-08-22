import torch 
import torch.nn as nn 
import transformers
import torch.nn.functional as F 

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, DeiTConfig, DeiTModel 




class ImageEmbedding(nn.Module):
    def __init__(self, output_size=768):
        super(ImageEmbedding, self).__init__()
        self.process = AutoImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.model = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        inputs = self.process(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs.to(device))

        image_embedding = outputs.last_hidden_state
        return image_embedding
    
    
    
class ImageEncoder(nn.Module):
    def __init__(self, d_model=1024):
        super(ImageEncoder, self).__init__()
        # this pretrained from hugging face, we can use ViT but it's too large
        self.process = AutoImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.model = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.fc = nn.Linear(self.model.config.hidden_size, d_model)
        #freeze params in model extractor
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        for param in self.model.parameters():
            param.requires_grad = False 
        
    
    def forward(self, img):
        inputs  = self.process(img, return_tensor = 'pt')
        with torch.no_grad() :
            outputs = self.model(**inputs.to(self.device))
        
        outs = self.fc(outputs)
        return  outs
    
