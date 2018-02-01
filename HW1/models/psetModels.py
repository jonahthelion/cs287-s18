import torch
import torch.nn as nn

class MNB(nn.Module):
    def __init__(self, V):
        super(MNB, self).__init__()

        self.V = V

        self.w_pos = nn.Embedding(V , 1 )
        self.w_neg = nn.Embedding(V , 1 )
