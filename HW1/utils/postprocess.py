import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
from .preprocess import text_to_img

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
        vis_windows = {}
        vis_windows['train_bce'] = vis.line(Y=torch.Tensor([float(train_l)]) , X=torch.Tensor([x_coord]), opts=dict(title='Train BCE'))
        vis_windows['val_bce'] = vis.line(Y=torch.Tensor([float(val_l)]) , X=torch.Tensor([x_coord]), opts=dict(title='Validation Accuracy'))
    else:
        vis.line(Y=torch.Tensor([float(train_l)]) , X=torch.Tensor([x_coord]), win=vis_windows['train_bce'], update='append', opts=dict(title='Train BCE'))
        if not val_l is None:
            vis.line(Y=torch.Tensor([float(val_l)]) , X=torch.Tensor([x_coord]), win=vis_windows['val_bce'], update='append', opts=dict(title='Validation Accuracy'))
    return vis_windows

def evaluate_model(model, val_iter, TEXT=None):
    all_actual, all_preds = [],[]

    for batch in val_iter:
        if TEXT is None:
            all_preds.append(model(batch.text.data).squeeze().cpu())
        else:
            imgs = text_to_img(batch.text.data, TEXT)
            print(imgs)
            all_preds.append(model(imgs).squeeze().cpu())
        all_actual.append(batch.label.data - 1)

    all_actual = Variable(torch.cat(all_actual))
    all_preds = torch.cat(all_preds)

    # binary cross entropy loss
    bce_l = F.binary_cross_entropy_with_logits(all_preds, all_actual.float())

    # accuracy and roc_auc
    all_preds = F.sigmoid(all_preds).data.numpy()
    all_actual = all_actual.data.numpy()
    roc_auc = metrics.roc_auc_score(all_actual, all_preds)
    acc = metrics.accuracy_score(all_actual, all_preds.round())

    return bce_l.data.numpy()[0], roc_auc, acc
