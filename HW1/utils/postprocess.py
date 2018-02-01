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
        vis_windows['train_bce'] = vis.line(Y=np.array([train_l]) , X=np.array([x_coord]))
    else:
        vis.line(Y=np.array([train_l]) , X=np.array([x_coord]), win=vis_windows['train_bce'], update='append')
