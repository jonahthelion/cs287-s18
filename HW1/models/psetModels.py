import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter
import torch.nn.functional as F
from sklearn import metrics

class MNB(nn.Module):
    def __init__(self, V, alpha, counts):
        super(MNB, self).__init__()

        self.V = V
        self.alpha = alpha
        self.counts = counts

        # initialize counts
        self.w_counts = torch.zeros(2, V) + alpha
        self.label_counts = torch.zeros(2)

    def train_sample(self, label, text):

        for phrase_ix in range(text.shape[1]):
            c = Counter(text[:,phrase_ix].numpy())
            for val in c:
                if not self.counts:
                    self.w_counts[label[phrase_ix], val] += 1
                else:
                    self.w_counts[label[phrase_ix], val] += c[val]
            self.label_counts[label[phrase_ix]] += 1

    def postprocess(self):
        self.w = nn.Linear(self.V, 1)
        self.w.weight.data = ( (self.w_counts[1] / self.w_counts[1].sum()) / (self.w_counts[0] / self.w_counts[0].sum()) ).log().view(1, -1).cuda() 
        self.w.bias.data = torch.Tensor([self.label_counts[1] / self.label_counts[0]]).log().cuda()

    # text is batch.text.data, shoud return cuda variable
    def forward(self, text):
        word_vecs = torch.zeros(text.shape[1], self.V)
        for phrase_ix in range(text.shape[1]):
            c = Counter(text[:,phrase_ix].numpy())
            for val in c:
                if not self.counts:
                    word_vecs[phrase_ix, val] += 1
                else:
                    word_vecs[phrase_ix, val] += c[val]
        return self.w(Variable(word_vecs.cuda()))


class LogReg(nn.Module):
    def __init__(self, V, counts):
        super(LogReg, self).__init__()

        self.V = V
        self.counts = counts

        self.w = nn.Linear(V, 1)
        self.w.weight.data.zero_()

    def forward(self, text):
        word_vecs = torch.zeros(text.shape[1], self.V)
        for phrase_ix in range(text.shape[1]):
            c = Counter(text[:,phrase_ix].numpy())
            for val in c:
                if not self.counts:
                    word_vecs[phrase_ix, val] += 1
                else:
                    word_vecs[phrase_ix, val] += c[val]
        return self.w(Variable(word_vecs).cuda())



class CBOW(nn.Module):
    def __init__(self, V, embed, pool):
        super(CBOW, self).__init__()

        self.V = V
        self.pool = pool
        self.embed = nn.Embedding(V, 300)
        self.embed.weight.data = embed

        self.w = nn.Sequential(

            nn.Linear(300, 1),
            )

    def forward(self, text):
        embeds = self.embed(Variable(text.cuda()))
        if self.pool == 'sum':
            return self.w(embeds.sum(0))
        if self.pool == 'mean':
            return self.w(embeds.mean(0))
        if self.pool == 'max':
            return self.w(embeds.max(0)[0])




class Conv(nn.Module):
    def __init__(self, V, embed):
        super(Conv, self).__init__()

        self.V = V
        self.embed = nn.Embedding(V, 300)
        self.embed.weight.data = embed

        self.w3 = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1))
        self.w4 = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=100, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1))
        self.w5 = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=100, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1))


    def forward(self, text):
        embeds = self.embed(Variable(text.cuda())).permute(1,2,0)
        out3 = self.w3(embeds)
        out4 = self.w4(embeds)
        out5 = self.w5(embeds)

        return 






