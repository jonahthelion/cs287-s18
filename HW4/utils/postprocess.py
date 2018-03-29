import torch

def vis_display(vis, vis_windows, x_coord, train_l, MAP):
    if vis_windows is None:
        vis_windows = {}
        vis_windows['train_ce'] = vis.line(Y=torch.Tensor([float(2**train_l)]), X=torch.Tensor([x_coord]), opts=dict(title='Train XENT'))
        vis_windows['val_ce'] = vis.line(Y=torch.Tensor([float(2**MAP)]), X=torch.Tensor([x_coord]), opts=dict(title='Train KL'))
    else:
        vis.line(Y=torch.Tensor([float(2**train_l)]), X=torch.Tensor([x_coord]), win=vis_windows['train_ce'], update='append', opts=dict(title='Train XENT'))
        vis.line(Y=torch.Tensor([float(2**MAP)]), X=torch.Tensor([x_coord]), win=vis_windows['val_ce'], update='append', opts=dict(title='Train KL'))
    return vis_windows