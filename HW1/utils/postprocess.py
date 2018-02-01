import torch

def print_important(TEXT, bad_vals, bad_ixes, good_vals, good_ixes):
    print('BAD')
    for val,ix in zip(bad_vals, bad_ixes):
        print(TEXT.vocab.itos[ix], ' ', val)
    print ('\n', 'GOOD')
    for val,ix in zip(good_vals, good_ixes):
        print(TEXT.vocab.itos[ix], ' ', val)
