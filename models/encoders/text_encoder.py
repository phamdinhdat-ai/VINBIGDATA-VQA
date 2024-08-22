import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import AutoTokenizer, DistilBertModel
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, DeiTModel, GPT2Tokenizer, GPT2Model

class QuesEmbedding(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(QuesEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        # Set a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Option 1: Using EOS token as PAD
            # or
            # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Option 2: Adding a new PAD token

        self.gpt = GPT2Model.from_pretrained("openai-community/gpt2")
        self.lstm = nn.LSTM(input_size, output_size, batch_first=True)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    def forward(self, ques):
        tokenized_input = self.tokenizer(
            ques,
            return_tensors='pt',
            padding='max_length',
            max_length=74,
            truncation=True
        )
        ques = self.gpt(**tokenized_input.to(self.device)).last_hidden_state
        _, (h, _) = self.lstm(ques)
        return h.squeeze(0)






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

        
        

        
class AnsEmbedding(nn.Module):
    def __init__(self, input_size=768):
        super(AnsEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        # Set a padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Option 1: Using EOS token as PAD
            # or
            # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Option 2: Adding a new PAD token

        self.gpt2_model = GPT2Model.from_pretrained("openai-community/gpt2")
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    def forward(self, ans): 
        tokenized_input = self.tokenizer(
            ans,
            return_tensors='pt',
            padding='max_length',
            max_length=17,
            truncation=True,
            return_attention_mask=False
        )

        outputs = self.gpt2_model(**tokenized_input.to(device))
        hidden_states = outputs.last_hidden_state

        return tokenized_input['input_ids'], hidden_states