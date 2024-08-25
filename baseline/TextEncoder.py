import torch 
import torch.nn as nn 
import nltk
from nltk.tokenize import word_tokenize
from .utils import build_vocab, tokenize_and_convert_to_indices

class BaseTextEncoder(nn.Module):
    '''Use NLTK to tokenize and LSTM for embedding'''
    def __init__(self, vocab_size, embedding_dim=768, hidden_size=512, max_length=74):
        super(BaseTextEncoder, self).__init__()
        nltk.download('punkt')  # Ensure NLTK's tokenizer is available
        
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # for params in self.parameters
        
    def forward(self, ques, word_to_idx=vocab):
        # Tokenize the input question using NLTK
        tokens = word_tokenize(ques.lower())  # Tokenize and convert to lowercase
        
        # Convert tokens to indices using the provided vocabulary
        token_indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
        
        # Limit the tokens to max_length and pad if necessary
        if len(token_indices) > self.max_length:
            token_indices = token_indices[:self.max_length]
        else:
            token_indices += [word_to_idx['<PAD>']] * (self.max_length - len(token_indices))

        # Convert to tensor and add batch dimension
        token_indices = torch.tensor(token_indices).unsqueeze(0).to(device)

        # Pass through the embedding layer
        embeddings = self.embedding(token_indices)
        
        # Pass the embeddings through the LSTM
        _, (h, _) = self.lstm(embeddings)
        return h.squeeze(0)
    
class AnsEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_size=1024, max_length=17):
        super(AnsEmbedding, self).__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#         self.to(self.device)
        
    def forward(self, ans, vocab=vocab):
        # Tokenize the input answer and convert to indices
        token_indices = tokenize_and_convert_to_indices(ans, vocab, self.max_length)
        token_indices = torch.tensor(token_indices).unsqueeze(0).to(device)  # Add batch dimension

        # Pass through the embedding layer
        embeddings = self.embedding(token_indices)
        
        # Pass the embeddings through the LSTM
        _, (h, _) = self.lstm(embeddings)
        
        # Return both token indices (vocabulary) and the LSTM hidden state
        return token_indices, h.squeeze(0)