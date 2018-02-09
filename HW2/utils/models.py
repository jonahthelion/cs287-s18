import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

class TriGram(nn.Module):
    def __init__(self, model_dict):
        super(TriGram, self).__init__()

        self.alpha = model_dict['alpha']
        assert(sum(self.alpha) == 1), self.alpha
        self.V = model_dict['max_size'] + 2

        self.unary_counts = torch.zeros(self.V)
        self.binary_counts = torch.zeros(self.V, self.V)
        self.tert_counts = torch.zeros(self.V, self.V, self.V)

    def train_predict(self, text):
        text = text.data.cpu().numpy().flatten('F')
        for i in range(2, text.shape[0]):
            if not text[i] == 3:
                self.unary_counts[text[i]] += 1
                self.binary_counts[text[i-1], text[i]] += 1
                self.tert_counts[text[i-2], text[i-1], text[i]] += 1

        return

    def postprocess(self):
        model.unary_counts = F.normalize(model.unary_counts, p=1, dim=0)
        model.binary_counts = F.normalize(model.binary_counts, p=1, dim=1)
        model.tert_counts = F.normalize(model.tert_counts, p=1, dim=2)

    def forward(self, text):
        text = text.data.cpu().numpy()
        probs = torch.zeros(text.shape[1], self.V)
        for i in range(text.shape[1]):
            col = text[:,i]
            
            probs[i] += self.alpha[2] * self.unary_counts
            probs[i] += self.alpha[1] * self.binary_counts[col[-1]]
            probs[i] += self.alpha[0] * self.tert_counts[col[-2], col[-1]]

        return Variable(probs.cuda())

