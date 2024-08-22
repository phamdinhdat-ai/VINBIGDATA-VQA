
import torch
import torch.nn as nn 
import torch.nn.functional as F 




class StackedAttentionNets(nn.Module):
    def __init__(self, d=768, k=512, dropout=True):
        super(StackedAttentionNets, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)
        self.gelu = nn.GELU()

    def forward(self, vi, vq):
        hi = self.ff_image(vi)
        hq = self.ff_ques(vq)
#         ha = F.tanh(hi + hq)
        ha = self.gelu(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha, dim=1)
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq.squeeze(1)
        return u
    
    
    

class AttentionFusion(nn.Module):
    def __init__(self, d_model = 1024, out_dim = 512, dropout = True):
        super(AttentionFusion, self).__init__()
        self.out_dim = out_dim
        self.d_model = d_model
        self.is_dropout = dropout
        self.ff_img = nn.Linear(d_model, out_dim )
        self.ff_text = nn.Linear(d_model, out_dim )
        self.ff_att = nn.Linear(d_model, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p = self.dropout) 
    def forward(self, en_img:torch.Tensor, en_text:torch.Tensor):
        h_img = self.ff_img(en_img)
        h_text = self.ff_text(en_text)
        h_all = F.tanh(h_img + h_text)
        h_all = self.gelu(h_all)
        if self.is_dropout: 
            h_all = self.dropout(h_all)
        h_att  = self.ff_att(h_all)
        pi = F.softmax(h_att, dim=1)
        vi_k = (pi.unsqueeze(dim=2) * en_img).sum(dim=1)
        u_k = vi_k + en_text.unsqueeze(1)
        return u_k
# class StackedAttentionNets(nn.Module):
#     def __init__(self, d=768, k=512, dropout=True):
#         super(StackedAttentionNets, self).__init__()
#         self.ff_image = nn.Linear(d, k)
#         self.ff_ques = nn.Linear(d, k)
#         if dropout:
#             self.dropout = nn.Dropout(p=0.5)
#         self.ff_attention = nn.Linear(k, 1)
#         self.gelu = nn.GELU()

#     def forward(self, vi, vq):
#         hi = self.ff_image(vi)
#         hq = self.ff_ques(vq)
# #         ha = F.tanh(hi + hq)
#         ha = self.gelu(hi + hq)
#         if getattr(self, 'dropout'):
#             ha = self.dropout(ha)
#         ha = self.ff_attention(ha).squeeze(dim=2)
#         pi = F.softmax(ha, dim=1)
#         vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
#         u = vi_attended + vq.squeeze(1)
#         return u