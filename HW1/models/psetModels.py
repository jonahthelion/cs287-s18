import torch
import torch.nn as nn
from collections import Counter

class MNB(nn.Module):
    def __init__(self, V, alpha):
        super(MNB, self).__init__()

        self.V = V
        self.alpha = alpha

        # initialize counts
        self.w_counts = torch.zeros(2, V) + alpha
        self.label_counts = torch.zeros(2)

    def train_sample(self, label, text):

        for phrase_ix in range(text.shape[1]):
            c = Counter(text[:,phrase_ix].numpy())
            for val in c:
                self.w_counts[label[phrase_ix], val] += 1
            self.label_counts[label[phrase_ix]] += 1

    def postprocess(self):
        self.w = nn.Linear(self.V, 1)
        self.w.weight.data = ( (self.w_counts[1] / self.w_counts[1].sum()) / (self.w_counts[0] / self.w_counts[0].sum()) ).log() 
        self.w.bias.data = (self.label_counts[1] / self.label_counts[0]).log()


