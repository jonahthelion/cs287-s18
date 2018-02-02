import torch

def print_important(w, TEXT, k):
    bad_vals, bad_ixes = torch.topk(w, k, largest=True)
    good_vals, good_ixes = torch.topk(w, k, largest=False)
    print('BAD')
    for val_ix,val,ix in zip(range(len(bad_vals)), bad_vals.numpy(), bad_ixes.numpy()):
        print(val_ix, TEXT.vocab.itos[ix], ' ', val)
    print ('\n', 'GOOD')
    for val_ix,val,ix in zip(range(good_vals), good_vals.numpy(), good_ixes.numpy()):
        print(val_ix,TEXT.vocab.itos[ix], ' ', val)
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