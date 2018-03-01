import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class noAttention(nn.Module):
    def __init__(self, model_dict, DE, EN):
        super(noAttention, self).__init__()

        self.Vg = len(DE.vocab)
        self.Ve = len(EN.vocab)
        self.D = model_dict['D']
        self.num_encode = model_dict['num_encode']

        self.embed = nn.Embedding(self.Vg, self.D)
        self.encode = nn.LSTM(self.D, self.D, self.num_encode)

    def train_predict(self, src):
        embeds = self.embed(src)
        return self.encode(embeds)[0,-1]