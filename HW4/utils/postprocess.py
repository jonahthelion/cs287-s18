import torch

def vis_display(vis, vis_windows, x_coord, train_l, MAP, sample_img, val_kl, val_recon):
    if vis_windows is None:
        vis_windows = {}
        vis_windows['train_ce'] = vis.line(Y=torch.Tensor([float(train_l)]), X=torch.Tensor([x_coord]), opts=dict(title='Train XENT', ytickmax=0.5, ytickmin=0))
        vis_windows['val_ce'] = vis.line(Y=torch.Tensor([float(MAP)]), X=torch.Tensor([x_coord]), opts=dict(title='Train KL', ytickmax=0.5, ytickmin=0))
        vis_windows['val_recon'] = vis.line(Y=torch.Tensor([float(val_recon)]), X=torch.Tensor([x_coord]), opts=dict(title='Validation XENT', ytickmax=0.5, ytickmin=0))
        vis_windows['val_kl'] = vis.line(Y=torch.Tensor([float(val_kl)]), X=torch.Tensor([x_coord]), opts=dict(title='Validation KL', ytickmax=0.5, ytickmin=0))
        vis_windows['imgs'] = vis.images(sample_img, nrow=4)
    else:
        vis.line(Y=torch.Tensor([float(train_l)]), X=torch.Tensor([x_coord]), win=vis_windows['train_ce'], update='append', opts=dict(title='Train XENT', ytickmax=0.5, ytickmin=0))
        vis.line(Y=torch.Tensor([float(MAP)]), X=torch.Tensor([x_coord]), win=vis_windows['val_ce'], update='append', opts=dict(title='Train KL', ytickmax=0.5, ytickmin=0))
        if not val_kl is None:
            vis.line(Y=torch.Tensor([float(val_recon)]), X=torch.Tensor([x_coord]), win=vis_windows['val_recon'], update='append', opts=dict(title='Validation XENT', ytickmax=0.5, ytickmin=0))
            vis.line(Y=torch.Tensor([float(val_kl)]), X=torch.Tensor([x_coord]), win=vis_windows['val_kl'], update='append', opts=dict(title='Validation KL', ytickmax=0.5, ytickmin=0))
            vis.images(sample_img, win=vis_windows['imgs'], nrow=4)
    return vis_windows

def get_validation_loss(model, val_loader):
    val_reconstruct = []; val_kl = [];
    for data_ix,(img,label) in enumerate(val_loader):
        mu, logvar = model.get_encoding(Variable(img).cuda())
        img_out = model.get_decoding(mu)

        l_reconstruct = F.binary_cross_entropy(img_out.view(-1,28*28), Variable(img).cuda().view(-1, 28*28), size_average=False)
        l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        val_reconstruct.append(l_reconstruct.data.cpu()[0])
        val_kl.append(l_kl.data.cpu()[0])
    return np.mean(val_kl), np.mean(val_reconstruct) 