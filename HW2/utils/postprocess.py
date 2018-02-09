import torch
import torch.nn.functional as F
import torchtext

import numpy as np
from tqdm import tqdm

def evaluate(model, test_iter):
    all_actuals = []
    all_preds = []
    for batch in tqdm(test_iter):
        text = torch.stack([ batch.text[:,i] for i in range(batch.text.shape[1]) if batch.text.data[-1, i] != 3]).t() 

        preds = model.predict(text.cuda()).cpu()

        labels = text[-1]

        all_actuals.extend(labels)
        all_preds.extend(preds)
    all_actuals = torch.stack(all_actuals).squeeze()
    all_preds = torch.stack(all_preds)

    nll_l = F.nll_loss(all_preds.log(), all_actuals).data.numpy()[0]

    _,top_ranks = all_preds.data.topk(20,1)
    rs = ( top_ranks == (all_actuals.data.view(-1, 1)*torch.ones(1, 20).long()) )
    MAP = mean_average_precision(rs.numpy())

    return nll_l, MAP

def write_submission(model, fout, TEXT):
    test = torchtext.datasets.LanguageModelingDataset(path="PSET/input.txt",text_field=TEXT)
    samples = [row.rstrip().split(" ") if row_ix == 0 else row.rstrip().split(" ")[1:] for row_ix,row in enumerate(' '.join(test[0].text).split('___ <eos>'))][:-1]
    
    return samples

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


