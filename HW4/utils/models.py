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
                        nn.Conv2d(int(self.hidden/9), 8, 3, 1, 1),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(8, 8, 2, 2, output_padding=1),
                        nn.Conv2d(8, 8, 3, 1, 1),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(8, 8, 2, 2, 0),
                        nn.Conv2d(8, 16, 3, 1, 1),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(16, 16, 2, 2, 0),
                        nn.Conv2d(16, 1, 3, 1, 1),
                        nn.Tanh(),
                    )

        self.discrim = nn.Sequential(
                        nn.Conv2d(1, 16, 4, 2, 1),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16, 32, 4, 2, 1),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 32, 4, 2, 1),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 1, 4, 2, 1),
                    )
        # self.discrim = nn.Sequential(
        #                 nn.Linear(28*28, 50),
        #                 nn.LeakyReLU(),
        #                 nn.Linear(50, 50),
        #                 nn.LeakyReLU(),
        #                 nn.Linear(50, 1),
        #             )
    def get_decoding(self, z):
        return self.decoder(z.view(z.shape[0], int(self.hidden/9), 3, 3)).squeeze(1)

    def get_discrim(self, x):
        return self.discrim(x.view(-1, 1,28,28))

class CNNVAE(nn.Module):
    def __init__(self, args):
        super(CNNVAE, self).__init__()
        self.hidden = args.hidden
        self.encoder = nn.Sequential(
                        nn.Conv2d(1, 16, 4, 2, 1),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(16),
                        nn.Conv2d(16, 32, 4, 2, 1),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 32, 4, 2, 1),
                        nn.LeakyReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 2*self.hidden, 4, 2, 1),
                    )
        self.simple = nn.Linear(self.hidden, 36)
        self.decoder = nn.Sequential(
                        nn.Conv2d(4, 8, 3, 1, 1),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(8, 8, 2, 2, output_padding=1),
                        nn.Conv2d(8, 8, 3, 1, 1),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(8, 8, 2, 2, 0),
                        nn.Conv2d(8, 16, 3, 1, 1),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(16, 16, 2, 2, 0),
                        nn.Conv2d(16, 1, 3, 1, 1),
                    )

    def get_encoding(self, x):
        out = self.encoder(x.view(x.shape[0], 1,28,28)).squeeze(-1).squeeze(-1)
        return out[:,:self.hidden], out[:,self.hidden:]

    def get_decoding(self, z):
        return self.decoder(self.simple(z).view(z.shape[0],4,3,3)).squeeze(1)




