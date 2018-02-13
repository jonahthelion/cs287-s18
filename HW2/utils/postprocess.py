import torch
import torch.nn.functional as F
import torchtext
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm

def evaluate(model, test_iter):
    all_actuals = []
    all_preds = []
    for batch in tqdm(test_iter):
        all_preds.extend(model.predict(batch.text[:-1].cuda()).cpu())
        all_actuals.extend(batch.text[-1])


    all_actuals = torch.stack(all_actuals).squeeze()
    all_preds = torch.stack(all_preds)

    _,top_ranks = all_preds.data.topk(20,1)

    dotted = 1./torch.arange(1, 21)
    MAP = []
    for row_ix in range(len(all_actuals)):
        vals = (all_actuals[row_ix].data == top_ranks[row_ix]).float()
        MAP.append((vals * dotted).sum())

    return np.mean(MAP)

def write_submission(model, fout, TEXT):
    test = torchtext.datasets.LanguageModelingDataset(path="PSET/input.txt",text_field=TEXT)
    samples = [row.rstrip().split(" ") if row_ix == 0 else row.rstrip().split(" ")[1:] for row_ix,row in enumerate(' '.join(test[0].text).split('___ <eos>'))][:-1]
    samples = torch.stack([torch.Tensor([TEXT.vocab.stoi[ix] for ix in row]).long() for row in samples], 1)

    preds = model.predict ( Variable(samples).cuda() ).cpu()
    _,top_ranks = preds.data.topk(20,1)

    with open(fout, 'w') as writer:
        writer.write('id,word\n')
        for row_ix,row in enumerate(top_ranks):
            writer.write(str(row_ix+1) + ',')
            for counter,word_ix in enumerate(row):
                if counter != len(row) - 1:
                    writer.write(str(TEXT.vocab.itos[word_ix]) + ' ')
                else:
                    writer.write(str(TEXT.vocab.itos[word_ix]) + '\n')
    return samples, top_ranks


