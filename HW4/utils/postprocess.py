import torch

def vis_display(vis, vis_windows, x_coord, train_l, MAP):
    if vis_windows is None:
        vis_windows = {}
        vis_windows['train_ce'] = vis.line(Y=torch.Tensor([float(train_l)]), X=torch.Tensor([x_coord]), opts=dict(title='Train XENT', ytickmax=0.5, ytickmin=0))
        vis_windows['val_ce'] = vis.line(Y=torch.Tensor([float(MAP)]), X=torch.Tensor([x_coord]), opts=dict(title='Train KL', ytickmax=0.5, ytickmin=0))
    else:
        vis.line(Y=torch.Tensor([float(train_l)]), X=torch.Tensor([x_coord]), win=vis_windows['train_ce'], update='append', opts=dict(title='Train XENT', ytickmax=0.5, ytickmin=0))
        vis.line(Y=torch.Tensor([float(MAP)]), X=torch.Tensor([x_coord]), win=vis_windows['val_ce'], update='append', opts=dict(title='Train KL', ytickmax=0.5, ytickmin=0))
    return vis_windows