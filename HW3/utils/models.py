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
        self.num_decode = model_dict['num_decode']

        self.embed = nn.Embedding(self.Vg, self.D)
        self.encode = nn.LSTM(self.D, self.D, self.num_encode)
        self.decode = nn.LSTM(self.D, self.D, self.num_decode)

    def train_predict(self, src, trg):
        embeds = self.embed(src)
        encode = self.encode(embeds)[1]
        decode = self.decode(trg, encode)[0]

        return decode