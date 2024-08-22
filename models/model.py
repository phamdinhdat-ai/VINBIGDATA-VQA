import torch 
import torch.nn as nn 
import torch.nn.functional as F 


from fusion_layer import StackedAttentionNets
from encoders.backboneEncoder import ImageEmbedding, ImageEncoder
from encoders.text_encoder import QuesEmbedding, AnsEmbedding
from decoders.encoder_layers import Decoder, DecoderLayer, SequentialDecoder

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256

class VQAModel(nn.Module):

    def __init__(self, vocab_size=50257, output_size=768, d_model=768,
                 num_heads=4, ffn_hidden=2048, drop_prob=0.1, num_layers=5,
                 num_att_layers=1):
        super(VQAModel, self).__init__()
        self.image_model = ImageEmbedding(output_size=output_size).to(device)
        self.ques_model = QuesEmbedding(output_size=output_size).to(device)
        self.ans_model = AnsEmbedding().to(device)

        self.san_model = nn.ModuleList(
            [StackedAttentionNets(d=d_model, k=512, dropout=True)] * num_att_layers).to(device)

        self.decoder = Decoder(d_model, ffn_hidden, num_heads,
                               drop_prob, num_layers).to(device)

#         self.mlp = nn.Sequential(
#             nn.Dropout(p=0.3),
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, vocab_size))
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(d_model, vocab_size))

    def forward(self, images, questions, answers, mask, max_len=17):
        image_embeddings = self.image_model(images.to(device))
        image_embedds = image_embeddings.reshape(BATCH_SIZE, 768, -1).permute(0, 2, 1)

        ques_embeddings = self.ques_model(questions)
        ques_embedds = ques_embeddings.unsqueeze(1)

        for att_layer in self.san_model:
            att_embedds = att_layer(image_embedds.to(device), ques_embedds.to(device))

        #START DECODER
        ans_vocab, ans_embedds = self.ans_model(answers)

        x = ans_embedds # 16 * 17 * 768 muốn cho bằng len của toàn bộ dứ liệu
        y = att_embedds.unsqueeze(1).expand(-1, max_len, -1)
        if mask == False:
            out = self.decoder(x, y, mask=None).to(device)
        else:
            mask = torch.full([max_len, max_len] , float('-inf'))
            mask = torch.triu(mask, diagonal=1).to(device)

            out = self.decoder(x, y, mask).to(device)
        #END DECODER

        output_logits = self.mlp(out)
        return output_logits, ans_vocab