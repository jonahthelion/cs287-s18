import torch

def print_important(TEXT, bad_vals, bad_ixes, good_vals, good_ixes):
    print('BAD')
    for val,ix in zip(bad_vals, bad_ixes):
        print(TEXT.vocab.itos[ix], ' ', val)
    print ('\n', 'GOOD')
    for val,ix in zip(good_vals, good_ixes):
        print(TEXT.vocab.itos[ix], ' ', val)
    print('\n')

def vis_display(vis, vis_windows, train_l, x_coord):
    if vis_windows['train_bce'] is None:
        vis_windows['train_bce'] = vis.line(Y=torch.Tensor([float(train_l)]) , X=torch.Tensor([x_coord]))
    else:
        vis.line(Y=torch.Tensor([float(train_l)]) , X=torch.Tensor([x_coord]), win=vis_windows['train_bce'], update='append')
    return vis_windows