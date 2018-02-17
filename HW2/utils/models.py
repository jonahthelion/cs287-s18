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




# model_dict = {'max_size': 10001, # max is 10001
#                 'batch_size': 140, 
#                 'bptt_len': 6,
#                 'num_epochs': 5,

#                 'output': 'simple3.txt',

#                 'type': 'NN', 
#                 'lookback': 3,
#                 'd': 300,

#                 }

class NN(nn.Module):
    def __init__(self, model_dict):
        super(NN, self).__init__()
        self.V = model_dict['max_size'] + 2
        self.lookback = model_dict['lookback']
        assert(model_dict['bptt_len'] > model_dict['lookback']), model_dict
        self.d = model_dict['d']

        self.embed = nn.Embedding(self.V, self.d)

        self.head = nn.Sequential(
                nn.Linear(self.d*self.lookback, self.d*self.lookback),
                nn.ReLU(inplace=True),
                # nn.BatchNorm1d(self.d*self.lookback),
                nn.Dropout(p=0.3, inplace=True),

                nn.Linear(self.d*self.lookback, self.V),
            )

    def internal(self, text):
        embeds = self.embed(text[-self.lookback:])
        probs = torch.stack([torch.cat([row for row in embeds[:,i]]) for i in range(text.shape[1])])
        return self.head(probs)

    def train_predict(self, text):
        return self.internal(text[:-1])

    def postprocess(self):
        pass

    def predict(self, text):
        return F.softmax(self.internal(text), 1)

class Ensemb(object):
    def __init__(self, models, alpha):
        self.models = models
        self.alpha = alpha
        assert( abs(sum(self.alpha) - 1) < .001)

    def predict(self, text):
        outs = self.alpha[0] * self.models[0].predict(text)
        for i in range(1, len(self.alpha)):
            outs += self.alpha[i] * self.models[i].predict(text)
        return outs



model_dict = {'max_size': 10001, # max is 10001
                'batch_size': 70, 
                'bptt_len': 11,
                'num_epochs': 50,

                'output': 'simple3.txt',

                'type': 'NNLSTM', 
                'lookback': 10,
                'd': 300

                }
class NNLSTM(nn.Module):
    def __init__(self, model_dict):
        super(NNLSTM, self).__init__()
        self.V = model_dict['max_size'] + 2
        self.d = model_dict['d']
        self.lookback = model_dict['lookback']

        self.embed = nn.Embedding(self.V, self.d)

        self.lstm = torch.nn.LSTM(self.d, self.d,dropout=.2, num_layers=2)

        self.head = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(self.d*self.lookback),
            nn.Dropout(p=.2 , inplace=True),

            nn.Linear(self.d, self.V),
        )

    def internal(self, text):
        embeds = self.embed(text[-self.lookback:])
        probs = self.lstm(embeds)[1][0][-1]
        return self.head(probs)

    def train_predict(self, text):
        return self.internal(text[:-1])

    def predict(self, text):
        return F.softmax(self.internal(text), 1)

    def postprocess(self):
        pass









