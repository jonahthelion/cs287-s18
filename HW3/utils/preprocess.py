import torch
import torchtext
import spacy

# from .models import TriGram, NN, NNLSTM

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def get_data(model_dict):



    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    DE = torchtext.data.Field(tokenize=tokenize_de)
    EN = torchtext.data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD)

    MAX_LEN = 20
    train, val, test = torchtext.datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
        len(vars(x)['trg']) <= MAX_LEN)

    MIN_FREQ = 5
    DE.build_vocab(train.src, min_freq=MIN_FREQ)
    EN.build_vocab(train.trg, min_freq=MIN_FREQ)

    BATCH_SIZE = 32
    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=-1,
                                                      repeat=False, sort_key=lambda x: len(x.src))

    return train_iter, val_iter, DE, EN