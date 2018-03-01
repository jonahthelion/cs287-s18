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

        self.embedder = nn.Embedding(self.Vg, self.D)
        self.encoder = nn.LSTM(self.D, self.D, self.num_encode)
        self.decoder = nn.LSTM(self.D, self.D, self.num_decode)

        self.classifier = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Linear(self.D, self.Ve),
            )

    def get_encode(self, src):
        embedded = self.embedder(src)
        output, hidden = self.encoder(embedded, (Variable(torch.zeros(self.num_encode, src.shape[1], self.D).cuda()), Variable(torch.zeros(self.num_encode, src.shape[1], self.D).cuda())))
        return output, hidden

    def get_decode(self, trg, hidden):
        embedded = self.embedder(trg)
        decode = self.decoder(embedded, hidden)
        return self.classifier(decode[0]), decode[1]
