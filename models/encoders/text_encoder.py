import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import AutoTokenizer, DistilBertModel


class TextEncoder(nn.Module):
    def __init__(self, d_model, max_len = 128):
        super(TextEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token  = self.tokenizer.eos_token
        
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(self.model.config.output_hidden_states, self.d_model)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    def forward(self, text):
        
        tokenized_input = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_len,
            truncation=True
        )
        text_feas = self.model(**tokenized_input.to(self.device)).last_hidden_state
        out = self.dropout(text_feas)
        out = self.fc(out)
        return out

        