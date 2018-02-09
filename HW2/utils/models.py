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

    def train_predict(self, text):
        text = text.data.cpu().numpy().flatten()
        c = Counter(text)
        for key in c:
            self.unary_counts[key] += c[key]

        return