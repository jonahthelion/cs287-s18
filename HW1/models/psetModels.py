import torch
import torch.nn as nn
from torch.autograd import Variable
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
        self.w.weight.data = ( (self.w_counts[1] / self.w_counts[1].sum()) / (self.w_counts[0] / self.w_counts[0].sum()) ).log().view(1, -1) 
        self.w.bias.data = torch.Tensor([self.label_counts[1] / self.label_counts[0]]).log()

    def find_important_words(k):
        bad_vals, bad_ixes = torch.topk(self.w.weight, 0, k, largest=True)
        good_vals, good_ixes = torch.topk(self.w.weight, 0, k, largest=False)
        return bad_vals, bad_ixes, good_vals, good_ixes

    def forward(self, text):
        word_vecs = torch.zeros(text.shape[1], self.V)
        for phrase_ix in range(text.shape[1]):
            c = Counter(text[:,phrase_ix].numpy())
            for val in c:
                word_vecs[phrase_ix, val] += 1
        return self.w(Variable(word_vecs)).data

