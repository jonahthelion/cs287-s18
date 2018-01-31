import torchtext
from torchtext.vocab import Vectors, GloVe

TEXT = torchtext.data.Field()

LABEL = torchtext.data.Field(sequential=False)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')