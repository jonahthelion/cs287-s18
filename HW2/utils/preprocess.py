import torch
import torchtext

from .models import TriGram

def get_data(model_dict):
    TEXT = torchtext.data.Field()

    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path="PSET", 
        train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

    TEXT.build_vocab(train, max_size=model_dict['max_size'])

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test), batch_size=model_dict['batch_size'], device=-1, bptt_len=model_dict['bptt_len'], repeat=False)

    return train_iter, val_iter, test_iter, TEXT


def get_model(model_dict):
    print(model_dict)
    if model_dict['type'] == 'trigram':
        model = TriGram(model_dict)

    return model