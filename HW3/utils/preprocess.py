import torch
import torchtext
import spacy
from time import time
from torch.autograd import Variable
import pickle

from .models import noAttention, Attention

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def get_data(model_dict):
    print(model_dict)
    t0 = time()

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    DE = torchtext.data.Field(tokenize=tokenize_de)
    EN = torchtext.data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD)

    if not model_dict['fake']:
        MAX_LEN = 20
        train, val, test = torchtext.datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
            filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)

    if not model_dict['pickled_fields']:
        MIN_FREQ = 5
        DE.build_vocab(train.src, min_freq=MIN_FREQ)
        EN.build_vocab(train.trg, min_freq=MIN_FREQ)
    else:
        DE = pickle.load( open( "DEfield.p", "rb" ) )
        EN = pickle.load( open( "ENfield.p", "rb" ) )

    if not model_dict['fake']:
        BATCH_SIZE = 32
        train_iter, val_iter = torchtext.data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                          repeat=False, sort_key=lambda x: len(x.src))
    else:
        val_iter = None
        train_iter = [Datapoint()]

    t1 = time()
    print('done loading', t1-t0)
    return train_iter, val_iter, DE, EN

def get_model(model_dict, DE, EN):
    if model_dict['type'] == 'noAttention':
        model = noAttention(model_dict, DE, EN)
        model.cuda()
        return model
    if model_dict['type'] == 'Attention':
        model = Attention(model_dict, DE, EN)
        model.cuda()
        return model

class Datapoint(object):
    def __init__(self):
        self.src = Variable(torch.zeros(8, 32).long())
        self.trg = Variable(torch.ones(12, 32).long())
