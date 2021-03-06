import torch
import torchtext

from .models import TriGram, NN, NNLSTM

def get_data(model_dict):
    TEXT = torchtext.data.Field()

    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path="PSET", 
        train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

    TEXT.build_vocab(train, max_size=model_dict['max_size'])

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test), batch_size=model_dict['batch_size'], device=-1, bptt_len=model_dict['bptt_len'], repeat=False)

    val_iter = torchtext.data.BPTTIterator(val, train=False, batch_size=67, device=-1, bptt_len=11)

    return train_iter, val_iter, test_iter, TEXT


def get_model(model_dict):
    print(model_dict)
    if model_dict['type'] == 'trigram':
        model = TriGram(model_dict)

    if model_dict['type'] == 'NN':
        model = NN(model_dict)
        model.cuda()

    if model_dict['type'] == 'NNLSTM':
        model = NNLSTM(model_dict)
        model.cuda()

    return model