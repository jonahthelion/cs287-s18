import torchtext
from torchtext.vocab import Vectors, GloVe

import matplotlib.pyplot as plt

def get_data(chosen_model):
    batch_size = chosen_model['batch_size']


    TEXT = torchtext.data.Field()

    LABEL = torchtext.data.Field(sequential=False)

    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size, device=-1, repeat=False)

    if not 'embed_type' in chosen_model or chosen_model['embed_type'] is 'wiki':
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # 'glove.6B.300d' # vectors=Vectors('wiki.simple.vec', url=url))
    elif chosen_model['embed_type'] is 'glove':
        TEXT.vocab.load_vectors(vectors='glove.6B.300d')


    return TEXT, LABEL, train_iter, val_iter, test_iter

def text_to_img(text, TEXT):
    sample_ix = 0
    img_text = " ".join(TEXT.vocab.itos[ix] for ix in text[:,sample_ix])
    pinrt(img_text)

