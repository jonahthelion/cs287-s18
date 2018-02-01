import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter
import torch.nn.functional as F

class MNB(nn.Module):
    def __init__(self, V, alpha):
        super(MNB, self).__init__()

        self.V = V
        self.alpha = alpha

        # initialize counts
        self.w_counts = torch.zeros(2, V) + alpha
        self.label_counts = torch.zeros(2)

    def train_sample(self, label, text):

        for phrase_ix in range(text.shape[1]):
            c = Counter(text[:,phrase_ix].numpy())
            for val in c:
                self.w_counts[label[phrase_ix], val] += 1
            self.label_counts[label[phrase_ix]] += 1

    def postprocess(self):
        self.w = nn.Linear(self.V, 1)
        self.w.weight.data = ( (self.w_counts[1] / self.w_counts[1].sum()) / (self.w_counts[0] / self.w_counts[0].sum()) ).log().view(1, -1) 
        self.w.bias.data = torch.Tensor([self.label_counts[1] / self.label_counts[0]]).log()

    def find_important_words(self, k):
        bad_vals, bad_ixes = torch.topk(self.w.weight.data, k, largest=True)
        good_vals, good_ixes = torch.topk(self.w.weight.data, k, largest=False)
        return bad_vals.squeeze(), bad_ixes.squeeze(), good_vals.squeeze(), good_ixes.squeeze()

    def forward(self, text):
        word_vecs = torch.zeros(text.shape[1], self.V)
        for phrase_ix in range(text.shape[1]):
            c = Counter(text[:,phrase_ix].numpy())
            for val in c:
                word_vecs[phrase_ix, val] += 1
        return self.w(Variable(word_vecs)).data

    def evalu(self, train_iter):
        all_actual = []
        all_preds = []
        for epoch in range(1):
            for batch_num,batch in enumerate(train_iter):
                preds = self.forward(batch.text.data)
                preds = F.sigmoid(preds)
                all_actual.append(batch.label.data - 1)
                all_preds.append(preds)
        all_actual = torch.cat(all_actual).numpy()
        all_preds = torch.cat(all_preds).numpy()

        return all_actual, all_preds

    def submission(self, test_iter, fname):
        print ('saving to', fname)
        upload = []
        for batch in test_iter:
            probs = F.sigmoid(self.forward(batch.text.data)) + 1
            upload.extend(list(probs.numpy().round().astype(int).flatten()))
        with open(fname, 'w') as f:
            f.write('Id,Cat\n')
            for u_ix,u in enumerate(upload):
                f.write(str(u_ix) + ',' + str(u) + '\n')


class LogReg(nn.Module):
    def __init__(self, V):
        super(LogReg, self).__init__()

        self.V = V

        self.w = nn.Linear(V, 1)
        torch.nn.init.xavier_uniform(self.w.weight.data)
        torch.nn.init.constant(self.w.bias.data, 0.0)

    def forward(self, text):
        word_vecs = torch.zeros(text.shape[1], self.V)
        for phrase_ix in range(text.shape[1]):
            c = Counter(text[:,phrase_ix].numpy())
            for val in c:
                word_vecs[phrase_ix, val] += 1
        return self.w(Variable(word_vecs)).view(-1)  

    def train_sample(self, label, text, optimizer):
        optimizer.zero_grad()
        outs = self.forward(text)
        l = F.binary_cross_entropy_with_logits(outs, label)
        l.backward()
        optimizer.step()

        return l

    def evalu_loss(self, label, text):
        outs = self.forward(text)
        l = F.binary_cross_entropy_with_logits(outs, label)
        return l

    def submission(self, test_iter, fname):
        print ('saving to', fname)
        upload = []
        for batch in test_iter:
            probs = F.sigmoid(self.forward(batch.text.data)) + 1
            upload.extend(list(probs.data.numpy().round().astype(int).flatten()))
        with open(fname, 'w') as f:
            f.write('Id,Cat\n')
            for u_ix,u in enumerate(upload):
                f.write(str(u_ix) + ',' + str(u) + '\n')














