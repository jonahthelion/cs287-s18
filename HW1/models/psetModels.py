import torch
import torch.nn as nn

class MNB(nn.Module):
    def __init__(self, V):
        super(MNB, self).__init__()

        self.V = V

        # initialize counts
        self.w_pos = torch.zeros(V)
        self.w_neg = torch.zeros(V)

    def train_sample(label, text):
        large_label = label.view(1, -1) * Variable(torch.ones(text.shape[0], 1).long())


