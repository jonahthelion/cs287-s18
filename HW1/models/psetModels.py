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
    def __init__(self, V, embed):
        super(CBOW, self).__init__()

        self.V = V
        self.embed = nn.Embedding(V, 300)
        self.embed.weight.data = embed

        self.w = nn.Sequential(

            nn.Linear(50, 1),
            )

    def forward(self, text):
        embeds = self.embed(Variable(text.cuda()))
        return self.w(embeds.mean(0))




class Conv(nn.Module):
    def __init__(self, V, embed):
        super(Conv, self).__init__()

        self.V = V
        self.embed = nn.Embedding(V, 300)
        self.embed.weight.data = embed

        self.w = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 1, 0),

            nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),

            nn.AdaptiveMaxPool1d(1),

            # nn.Conv1d(500, 300, 1),
            # nn.BatchNorm1d(300),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Conv1d(300, 1, 1),
            )

    def forward(self, text):
        if text.shape[1] == 1:
            text = torch.stack([text[:,0], torch.zeros(text.shape[0]).long()], dim=1)
            embeds = torch.stack([torch.t(self.embed(text[:,i])) for i in range(text.shape[1])])[0].unsqueeze(0)
        else:
            embeds = torch.stack([torch.t(self.embed(text[:,i])) for i in range(text.shape[1])])

        return self.w(embeds).view(-1)    

    def evalu_loss(self, label, text):
        outs = self.forward(text)
        l = F.binary_cross_entropy_with_logits(outs, label)
        return l

    def evalu(self, train_iter):
        all_actual = []
        all_preds = []
        for epoch in range(1):
            for batch_num,batch in enumerate(train_iter):
                preds = self.forward(batch.text.data)
                preds = F.sigmoid(preds)
                all_actual.append(batch.label.data - 1)
                all_preds.append(preds.data)
        all_actual = torch.cat(all_actual).numpy()
        all_preds = torch.cat(all_preds).numpy()

        print(metrics.classification_report(all_actual, all_preds.round()))

        return all_actual, all_preds





