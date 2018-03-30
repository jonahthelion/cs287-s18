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
                        nn.Linear(50, 2*self.hidden),
                    )


        self.decoder = nn.Sequential(
                        nn.Linear(self.hidden, 50),
                        nn.ReLU(),
                        nn.Linear(50, 50),
                        nn.ReLU(),
                        nn.Linear(50, 28*28),
                    )

    def get_encoding(self, x):
        out = self.encoder(x.view(x.shape[0], 28*28))
        return out[:,:self.hidden], out[:,self.hidden:]

    def get_decoding(self, z):
        return self.decoder(z).view(z.shape[0], 28, 28)

class SimpleGAN(nn.Module):
    def __init__(self, args):
        super(SimpleGAN, self).__init__()
        self.hidden = args.hidden

        self.decoder = nn.Sequential(
                        nn.Linear(self.hidden, 50),
                        nn.LeakyReLU(),
                        nn.Linear(50, 50),
                        nn.LeakyReLU(),
                        nn.Linear(50, 28*28),
                        nn.Tanh(),
                    )

        self.discrim = nn.Sequential(
                        nn.Linear(28*28, 50),
                        nn.LeakyReLU(),
                        nn.Linear(50, 50),
                        nn.LeakyReLU(),
                        nn.Linear(50, 1),
                    )

    def get_decoding(self, z):
        return self.decoder(z).view(z.shape[0], 28, 28)

    def get_discrim(self, x):
        return self.discrim(x)

class CNNGAN(nn.Module):
    def __init__(self, args):
        super(CNNGAN, self).__init__()
        self.hidden = args.hidden

        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(args.hidden, 128, 2, 1, 0),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(128, 128, 3, 2, 0),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(128, 128, 3, 1, 0),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(128, 128, 1, 2, 0),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(128, 128, 4, 2, 0),
                        nn.Tanh(),
                    )

        self.discrim = nn.Sequential(
                        nn.Conv2d(1, 8, 2),
                        nn.LeakyReLU(),
                        nn.Conv2d(1, 8, 2),
                        nn.LeakyReLU(),
                        nn.Conv2d(1, 8, 2),
                        nn.LeakyReLU(),
                        nn.Conv2d(1, 8, 2),
                    )
    def get_decoding(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(z)

    def get_discrim(self, x):
        return self.discrim(x.view(-1, 1,28,28))






