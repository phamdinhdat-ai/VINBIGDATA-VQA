import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import AutoTokenizer, BertTokenizer, BertModel, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, d_model):
        super(TextEncoder, self).__init__()
        
        self.d_model = d_model
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        self.model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="flash_attention_2")


