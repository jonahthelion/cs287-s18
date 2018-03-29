import torch.nn as nn

class SimpleVAE(nn.Module):
    def __init__(self, args):
        super(SimpleVAE, self).__init__()
        self.hidden = args.hidden
        self.encoder = nn.Linear(28*28, 2*self.hidden)

        self.decoder = nn.Linear(2*self.hidden, 28*28)

    def get_encoding(self, x):
        flatx = x.view(x.shape[0], 28*28)
        out = self.encoder(flatx)
        return out[:,:self.hidden], out[:,self.hidden:]