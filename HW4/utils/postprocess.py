import torch

def vis_display(vis, vis_windows, x_coord, train_l, MAP, sample_img, val_kl, val_recon):
    if vis_windows is None:
        vis_windows = {}
        vis_windows['train_ce'] = vis.line(Y=torch.Tensor([float(train_l)]), X=torch.Tensor([x_coord]), opts=dict(title='Train XENT', ytickmax=0.5, ytickmin=0))
        vis_windows['val_ce'] = vis.line(Y=torch.Tensor([float(MAP)]), X=torch.Tensor([x_coord]), opts=dict(title='Train KL', ytickmax=0.5, ytickmin=0))
        vis_windows['val_recon'] = vis.line(Y=torch.Tensor([float(val_recon)]), X=torch.Tensor([x_coord]), opts=dict(title='Validation XENT', ytickmax=0.5, ytickmin=0))
        vis_windows['val_kl'] = vis.line(Y=torch.Tensor([float(val_kl)]), X=torch.Tensor([x_coord]), opts=dict(title='Validation KL', ytickmax=0.5, ytickmin=0))
        vis_windows['imgs'] = vis.images(sample_img)
    else:
        vis.line(Y=torch.Tensor([float(train_l)]), X=torch.Tensor([x_coord]), win=vis_windows['train_ce'], update='append', opts=dict(title='Train XENT', ytickmax=0.5, ytickmin=0))
        vis.line(Y=torch.Tensor([float(MAP)]), X=torch.Tensor([x_coord]), win=vis_windows['val_ce'], update='append', opts=dict(title='Train KL', ytickmax=0.5, ytickmin=0))
        vis.line(Y=torch.Tensor([float(val_recon)]), X=torch.Tensor([x_coord]), win=vis_windows['val_recon'], update='append', opts=dict(title='Validation XENT', ytickmax=0.5, ytickmin=0))
        vis.line(Y=torch.Tensor([float(val_kl)]), X=torch.Tensor([x_coord]), win=vis_windows['val_kl'], update='append', opts=dict(title='Validation KL', ytickmax=0.5, ytickmin=0))
        vis.images(sample_img, win=vis_windows['imgs'])
    return vis_windows