import torch
from torch.utils import data as torchdata
import os

import spacy
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')
spacy_fr = spacy.load('fr')

def tokenize_de(text):
    return ['<de>'] + [tok.text for tok in spacy_de.tokenizer(text.strip('\n'))] + ['</de>']

def tokenize_en(text):
    return ['<en>'] + [tok.text for tok in spacy_en.tokenizer(text.strip('\n'))] + ['</en>'] 

def tokenize_fr(text):
    return ['<fr>'] + [tok.text for tok in spacy_fr.tokenizer(text.strip('\n'))] + ['</fr>']

class Multi30kLoader(torchdata.Dataset):
    def __init__(self, args, data_type):
        self.root = args.d
        self.data_type = data_type

        self.langs = ['en', 'de', 'fr']
        self.token_func = {'en': tokenize_en, 'fr': tokenize_fr, 'de': tokenize_de}
        self.data = {}
        for lang in self.langs:
            self.data[lang] = [self.token_func[lang](row) for row in open(os.path.join(self.root, 'data/task1/raw/' + self.data_type + '.' + lang), 'r').readlines()]
        self.TEXT = None

    def __getitem__(self, ix):
        out = {lang: torch.LongTensor([self.TEXT.stoi[lang, word] if (lang, word) in self.TEXT.stoi else self.TEXT.stoi['<unk>'] for word in self.data[lang][ix]]) for lang in self.langs}
        out = {lang: torch.cat((out[lang], torch.zeros(15-len(out[lang])).long()-2), 0) for lang in out}
        return out

    def __len__(self):
        return len(self.data)

class Dictionary(object):
    def __init__(self, trainloader, testloader):
        self.itos = {}
        self.stoi = {}
        i = 0
        for lang in trainloader.langs:
            for row in trainloader.data[lang]:
                for word in row:
                    if not (lang, word) in self.stoi:
                        self.itos[i] = lang, word
                        self.stoi[lang, word] = i
                        i += 1
        self.itos[i] = '<unk>'
        self.stoi['<unk>'] = i

def get_data(args):
    trainloader = Multi30kLoader(args, 'train')
    valloader = Multi30kLoader(args, 'val')
    TEXT = Dictionary(trainloader, valloader)
    trainloader.TEXT = TEXT; valloader.TEXT = TEXT;

    return trainloader, valloader, TEXT