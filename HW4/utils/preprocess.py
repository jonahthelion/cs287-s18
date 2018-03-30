import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.models import SimpleVAE, SimpleGAN, CNNGAN

def get_data(args, bern=True):
    print('Loading Data...')
    train_dataset = datasets.MNIST(root='./data/',
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=False)
    test_dataset = datasets.MNIST(root='./data/',
                               train=False, 
                               transform=transforms.ToTensor())

    torch.manual_seed(3435)
    if bern:
        train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    else:
        train_img = torch.stack([d[0] for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])
    if bern:
        test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    else:
        test_img = torch.stack([d[0] for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])

    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()
    train_img = train_img[:-10000]
    train_label = train_label[:-10000]


    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)
    BATCH_SIZE = 100
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader

def get_model(args):
    if args.model == 'Simple':
        return SimpleVAE(args).cuda()
    elif args.model == 'SimpleGAN':
        return SimpleGAN(args).cuda()
    elif args.model == 'CNNGAN':
        return CNNGAN(args).cuda()