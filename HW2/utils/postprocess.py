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

    nll_l = F.nll_loss(all_preds.log(), all_actuals).data.numpy()[0]

    _,top_ranks = all_preds.data.topk(20,1)

    dotted = 1./torch.arange(1, 21)
    MAP = []
    for row_ix in range(len(all_actuals)):
        vals = (all_actuals[row_ix].data == top_ranks[row_ix]).float()
        MAP.append((vals * dotted).sum())


    return nll_l, MAP

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

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


