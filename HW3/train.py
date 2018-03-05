import torchtext
import torch
from torchtext.vocab import Vectors
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
import visdom
import matplotlib.pyplot as plt

from utils.preprocess import get_data, get_model
from utils.postprocess import vis_display, evaluate

# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

#############################

model_dict = {'type': 'Attention',
                'D': 300,
                'num_encode': 2,
                'num_decode': 2,
                'num_epochs': 50,
                'fake': True,
                'pickled_fields': True}

train_iter, val_iter, DE, EN = get_data(model_dict)

###########
model = torch.load('Attention2.p')
model.encoder.flatten_parameters()
model.decoder.flatten_parameters()
model.eval()
fname = 'PSET/source_test.txt'
answers = []
with open(fname, 'rb') as reader:
    for break_ix, line in tqdm(enumerate(reader)):
        src = Variable(torch.Tensor([DE.vocab.stoi[s] for s in line.decode('utf-8').strip('\n').split(' ')]).long().unsqueeze(1))
        output, hidden = model.get_encode(src.cuda())

        actual_sentences = []; actual_scores = [];
        poss_sentences = Variable(torch.Tensor([[2]]).long().cuda())
        poss_scores = [0]
        while len(actual_sentences) < 100:
            poss_hidden = torch.stack([output[:,0] for _ in range(poss_sentences.shape[1])], 1), (torch.stack([hidden[0][:,0] for _ in range(poss_sentences.shape[1])], 1), torch.stack([hidden[1][:,0] for _ in range(poss_sentences.shape[1])], 1)) 
            preds = model.get_decode(poss_sentences, poss_hidden)

            new_sentences = []; new_scores = [];
            for i in range(poss_sentences.shape[1]):
                best_pred_vals, best_pred_ixes = preds[0][-1, i].topk(5)
                for val_ix in range(len(best_pred_ixes)):
                    if best_pred_ixes[val_ix].cpu().data.numpy()[0] == 3:
                        actual_sentences.append(torch.cat((poss_sentences[:,i], best_pred_ixes[val_ix] )))
                        actual_scores.append(poss_scores[i] + best_pred_vals[val_ix])
                    else:
                        new_sentences.append(torch.cat((poss_sentences[:,i], best_pred_ixes[val_ix] )))
                        new_scores.append(poss_scores[i] + best_pred_vals[val_ix])
            poss_sentences = torch.stack(new_sentences, 1); poss_scores = torch.stack(new_scores)
            if poss_sentences.shape[1] > 100:
                best_ixes = poss_scores.topk(100,0)[1].squeeze(1).data
                poss_sentences = torch.stack([poss_sentences[:,ix] for ix in best_ixes], 1)
                poss_scores = poss_scores[best_ixes]
        actual_scores = torch.stack(actual_scores)
        best_ixes = actual_scores.topk(100,0)[1].squeeze(1).data
        actual_sentences = [actual_sentences[ix] for ix in best_ixes]
        actual_scores = actual_scores[best_ixes]

        ### printing the translations
        print(break_ix, ' '.join([EN.vocab.itos[actual_sentences[0].data[c]] for c in range(len(actual_sentences[0]))])) 

        ### printing the attention
        all_weights = []
        trg = actual_sentences[0].unsqueeze(1)
        encoding = output, hidden
        current_hidden = encoding[1]
        if encoding[0].shape[0] <20:
            encoding_hist = torch.cat((encoding[0], Variable(torch.zeros(20 - encoding[0].shape[0], encoding[0].shape[1], encoding[0].shape[2]).cuda())))
        else:
            encoding_hist = encoding[0]

        for i in range(trg.shape[0]):
            weights = F.softmax(model.attn(torch.cat((model.embedder(trg[i].unsqueeze(0)), current_hidden[0][-1].unsqueeze(0)), 2)), 2)
            feat_hist = torch.bmm(encoding_hist.permute(1, 2, 0), weights.permute(1, 2, 0))
            output, current_hidden = model.decoder(torch.cat((feat_hist.permute(2, 0, 1), model.embedder(trg[i].unsqueeze(0))), 2), current_hidden)
            all_weights.append(weights)
        all_weights = torch.stack(all_weights)
        fig = plt.figure()
        plt.imshow(all_weights.data.cpu().numpy())
        plt.tight_layout()
        plt.savefig('att_fig.pdf')
        plt.close(fig)
        assert False

        # fill answers
        answers.append([[EN.vocab.itos[ans.data[c]] for c in range(1,4)] if len(ans)>4 else ['<unk>','<unk>','<unk>'] for ans in actual_sentences])

with open('kaggle4.txt', 'w') as writer:
    writer.write('id,word\n')
    for li_ix,line in enumerate(answers):
        out = ''
        for li in line:
            out += '|'.join(li)
            out += ' '
        out = out.replace("\"", "<quote>").replace(",", "<comma>")
        out = str(li_ix+1) + ',' + out[:-1] + '\n'
        writer.write(out)
assert False
##########


#model = get_model(model_dict, DE, EN)
model = torch.load('Attention2.p')

optimizer = torch.optim.Adam(model.parameters(), lr=.0008, weight_decay=1e-4, betas=(.9, .999))

for epoch in range(model_dict['num_epochs']):
    for batch_num,batch in enumerate(train_iter):
        model.train()
        optimizer.zero_grad()

        encoding = model.get_encode(batch.src.cuda())
        print('THING', batch.trg.shape, encoding[0].shape, encoding[1][0].shape, encoding[1][1].shape)
        output, hidden = model.get_decode(batch.trg.cuda(), encoding)
        loss = F.cross_entropy(output[:-1].view(-1, output.shape[-1]), batch.trg.cuda()[1:].view(-1), ignore_index=1)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        if batch_num % 100 == 0:
            loss_t = loss.data.cpu().numpy()[0]
            loss_v = None
            if batch_num % 300 == 0:
                model.eval()
                loss_v = evaluate(model, val_iter)
                print(epoch, batch_num, 'Validate:', loss_v, 'Train:', loss_t)
            vis_windows = vis_display(vis, vis_windows, epoch + batch_num/float(len(train_iter)), loss_t, loss_v)



