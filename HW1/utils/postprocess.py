import torch

def print_important(TEXT, bad_vals, bad_ixes, good_vals, good_ixes):
    print('BAD')
    for val,ix in zip(bad_vals, bad_ixes):
        print(TEXT.vocab.itos[ix], ' ', val)
    print ('\n', 'GOOD')
    for val,ix in zip(good_vals, good_ixes):
        print(TEXT.vocab.itos[ix], ' ', val)
    print('\n')

def vis_display(vis, vis_windows, train_l, x_coord, val_l=None):
    if vis_windows is None:
        vis_windows['train_bce'] = vis.line(Y=torch.Tensor([float(train_l)]) , X=torch.Tensor([x_coord]), opts=dict(title='Train BCE'))
        vis_windows['val_bce'] = vis.line(Y=torch.Tensor([float(val_l)]) , X=torch.Tensor([x_coord]), opts=dict(title='Validation BCE'))
    else:
        vis.line(Y=torch.Tensor([float(train_l)]) , X=torch.Tensor([x_coord]), win=vis_windows['train_bce'], update='append', opts=dict(title='Train BCE'))
        if not val_l is None:
            vis.line(Y=torch.Tensor([float(val_l)]) , X=torch.Tensor([x_coord]), win=vis_windows['val_bce'], update='append', opts=dict(title='Validation BCE'))
    return vis_windows