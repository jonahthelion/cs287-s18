import torch
import torch.nn as nn

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
        for i in range(text.shape[0] - 2):
            self.unary_counts[text[i]] += 1
            self.binary_counts[text[i], text[i+1]] += 1
            self.tert_counts[text[i], text[i+1], text[i+2]] += 1

        return