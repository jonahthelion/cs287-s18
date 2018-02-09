import torch
import torch.nn as nn

class TriGram(nn.Module):
    def __init__(self, model_dict):
        super(TriGram, self).__init__()

        self.alpha = model_dict['alpha']
        assert(sum(self.alpha) == 1), self.alpha
