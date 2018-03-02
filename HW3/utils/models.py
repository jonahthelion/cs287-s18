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
                            nn.ReLU(),
                            nn.Linear(self.D, self.Ve),
            )

    def get_encode(self, src):
        embedded = self.embedder(torch.stack([src[len(src) -1 - i] for i in range(len(src)) ]))
        output, hidden = self.encoder(embedded)
        return output, hidden

    def get_decode(self, trg, encoding):
        embedded = self.embedder(trg)
        decode = self.decoder(embedded, encoding[1])
        return self.classifier(decode[0]), decode[1]

class Attention(nn.Module):
    def __init__(self, model_dict, DE, EN):
        super(Attention, self).__init__()

        self.Vg = len(DE.vocab)
        self.Ve = len(EN.vocab)
        self.D = model_dict['D']
        self.num_encode = model_dict['num_encode']
        self.num_decode = model_dict['num_decode']

        self.embedder = nn.Embedding(self.Vg, self.D)
        self.encoder = nn.LSTM(self.D, self.D, self.num_encode)
        self.decoder = nn.LSTM(self.D, self.D, self.num_decode)

        self.classifier = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(self.D, self.Ve),
            )

    def get_encode(self, src):
        embedded = self.embedder(torch.stack([src[len(src) -1 - i] for i in range(len(src)) ]))
        output, hidden = self.encoder(embedded)
        return output, hidden

    def get_decode(self, trg, encoding):
        pass
