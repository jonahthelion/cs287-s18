import torch.nn as nn
import torch.nn.functional as F

class SimpleVAE(nn.Module):
    def __init__(self, args):
        super(SimpleVAE, self).__init__()
        self.hidden = args.hidden
        self.encoder = nn.Sequential(
                        nn.Linear(28*28, 50),
                        nn.ReLU(),
                        nn.Linear(50, 50),
                        nn.ReLU(),
                        nn.Linear(50, self.hidden),
                    )


        self.decoder = nn.Sequential(
                        nn.Linear(self.hidden, 50),
                        nn.ReLU(),
                        nn.Linear(50, 508),
                        nn.ReLU(),
                        nn.Linear(50, 28*28),
                    )

    def get_encoding(self, x):
        out = self.encoder(x.view(x.shape[0], 28*28))
        return out[:,:self.hidden], F.softplus(out[:,self.hidden:])

    def get_decoding(self, z):
        return self.decoder(z).view(z.shape[0], 28, 28)