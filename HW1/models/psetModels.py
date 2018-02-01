import torch
import torch.nn as nn
from collections import Counter

class MNB(nn.Module):
    def __init__(self, V):
        super(MNB, self).__init__()

        self.V = V

        # initialize counts
        self.w_counts = [torch.zeros(V), torch.zeros(V)]
        self.label_counts = [0, 0]

    def train_sample(self, label, text):
        large_label = label.view(1, -1) * torch.ones(text.shape[0], 1).long()

        for lab in [0,1]:
            c = Counter(text[large_label==lab].numpy())
            for val in c.keys():
                self.w_counts[lab][val] += c[val]
            self.label_counts[lab] += (label == lab).sum()

