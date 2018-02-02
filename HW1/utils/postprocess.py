import torch
import torch.nn.functional as F
from torch.autograd import Variable

def print_important(w, TEXT, k):
    bad_vals, bad_ixes = torch.topk(w, k, largest=True)
    good_vals, good_ixes = torch.topk(w, k, largest=False)
    print('BAD')
    for val_ix,val,ix in zip(range(len(bad_vals)), bad_vals.numpy(), bad_ixes.numpy()):
        print(val_ix, TEXT.vocab.itos[ix], ' ', val)
    print ('\n', 'GOOD')
    for val_ix,val,ix in zip(range(len(good_vals)), good_vals.numpy(), good_ixes.numpy()):
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

def evaluate_model(model, val_iter):
    all_actual, all_preds = [],[]

    for batch in val_iter:
        all_preds.append(model(batch.text.data).squeeze(0))
        all_actual.append(batch.label.data - 1)

    all_actual = Variable(torch.cat(all_actual).cuda())
    all_preds = torch.cat(all_preds)

    # binary cross entropy loss
    bce_l = F.binary_cross_entropy_with_logits(all_preds, all_actual)

    # accuracy
    all_preds = F.sigmoid(all_preds).data.cpu().numpy()
    all_actual = all_actual.data.cpu().numpy()

    print(all_preds.shape, all_actual.shape)
