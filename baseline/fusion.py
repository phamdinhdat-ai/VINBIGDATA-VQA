import torch
import torch.nn as nn

class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()
    def forward(self, image_embed, text_embed):
#         image_embed = image_embed.unsqueeze(2)
#         text_embed = text_embed.unsqueeze(2)
        x = torch.cat((image_embed, text_embed), dim=1).to(device)
        return x