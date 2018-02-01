import torchtext
from torchtext.vocab import Vectors, GloVe

def get_data(batch_size):

    TEXT = torchtext.data.Field()

    LABEL = torchtext.data.Field(sequential=False)

    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=(batch_size,batch_size,10), device=-1, repeat=False)

    return TEXT, LABEL, train_iter, val_iter, test_iter