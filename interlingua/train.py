from torchtext import data
from torchtext import datasets
import spacy

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')
spacy_fr = spacy.load('fr')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

BOS_WORD = '<s>'
EOS_WORD = '</s>'
EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD)
DE = data.Field(tokenize=tokenize_de, init_token = BOS_WORD, eos_token = EOS_WORD)
FR = data.Field(tokenize=tokenize_fr, init_token = BOS_WORD, eos_token = EOS_WORD)

train, val, test = datasets.Multi30k.splits(exts=('.de', '.en', '.fr'), fields=(DE, EN, FR))