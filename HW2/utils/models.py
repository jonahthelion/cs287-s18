import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import Counter
from tqdm import tqdm


# model_dict = {'max_size': 10001, # max is 10001
#                 'batch_size': 41, 
#                 'bptt_len': 32,
#                 'num_epochs': 1,

#                 'output': 'simple3.txt',

#                 'type': 'trigram', 
#                 'alpha': [0.4306712668382596, 0.4897915705677378, 0.07953716259400256],

#                 # 'type': 'NN',
#                 }

class TriGram(nn.Module):
    def __init__(self, model_dict):
        super(TriGram, self).__init__()

        self.alpha = model_dict['alpha']
        assert(abs(sum(self.alpha) - 1) < .001), self.alpha
        assert(model_dict['num_epochs'] == 1), model_dict['num_epochs']
        self.V = model_dict['max_size'] + 2

        self.unary_counts = torch.zeros(self.V)
        self.binary_counts = torch.zeros(self.V, self.V)
        self.tert_counts = {}

    def train_predict(self, text):
        text = text.data.cpu().numpy().flatten('F')
        for i in range(2, text.shape[0]):
            self.unary_counts[text[i]] += 1
            self.binary_counts[text[i-1], text[i]] += 1
            if not (text[i-2], text[i-1]) in self.tert_counts:
                self.tert_counts[text[i-2], text[i-1]] = torch.zeros(self.V)
            self.tert_counts[text[i-2], text[i-1]][text[i]] += 1

        return

    def postprocess(self):
        self.unary_counts = F.normalize(self.unary_counts, p=1, dim=0)
        self.binary_counts = F.normalize(self.binary_counts, p=1, dim=1)
        for key in tqdm(self.tert_counts):
            self.tert_counts[key] = F.normalize(self.tert_counts[key], p=1, dim=0)

    # input is batch.text.cuda() (cuda Variable), output is cuda variable (Nbatch, V)
    def predict(self, text):
        text = text.data.cpu().numpy()
        probs = torch.zeros(text.shape[1], self.V)
        for i in range(text.shape[1]):
            col = text[:,i]

            probs[i] += self.alpha[2] * self.unary_counts

            if self.binary_counts[col[-1]].sum() > .1:
                probs[i] += self.alpha[1] * self.binary_counts[col[-1]]
            else:
                probs[i] += self.alpha[1] * self.unary_counts

            if (col[-2], col[-1]) in self.tert_counts:
                probs[i] += self.alpha[0] * self.tert_counts[col[-2], col[-1]]
            else:
                if self.binary_counts[col[-1]].sum() > .1:
                    probs[i] += self.alpha[0] * self.binary_counts[col[-1]]
                else:
                    probs[i] += self.alpha[0] * self.unary_counts

        return Variable(probs.cuda())




model_dict = {'max_size': 10001, # max is 10001
                'batch_size': 80, 
                'bptt_len': 4,
                'num_epochs': 5,

                'output': 'simple3.txt',

                'type': 'NN', 
                }

class NN(nn.Module):
    def __init__(self, model_dict):
        super(NN, self).__init__()
        self.V = model_dict['max_size'] + 2

        self.embed = nn.Embedding(self.V, 300)

        self.head = nn.Sequential(
                nn.Linear(300*3, self.V),
            )

    def train_predict(self, text):
        embeds = self.embed(text[-4:-1])
        probs = torch.stack([torch.cat([row for row in embeds[:,i]]) for i in range(text.shape[1])])
        return self.head(probs)

    def postprocess(self):
        pass

    def predict(self, text):
        return F.softmax(self.train_predict(text), 1)









