import torchtext
import torch
from torchtext.vocab import Vectors
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
import visdom


# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

#############################

