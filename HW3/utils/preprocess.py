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

    print(train.fields)
    print(len(train))
    print(vars(train[0]))