import torchtext
from torchtext.vocab import Vectors, GloVe

import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import textwrap
from torchvision import transforms
import torch
from torch.autograd import Variable

font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 14)
normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])

def get_data(chosen_model):
    batch_size = chosen_model['batch_size']


    TEXT = torchtext.data.Field()

    LABEL = torchtext.data.Field(sequential=False)

    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size, device=-1, repeat=False)

    if not 'embed_type' in chosen_model or chosen_model['embed_type'] is 'wiki':
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url)) # 'glove.6B.300d' # vectors=Vectors('wiki.simple.vec', url=url))
    elif chosen_model['embed_type'] is 'glove':
        TEXT.vocab.load_vectors(vectors='glove.6B.300d')


    return TEXT, LABEL, train_iter, val_iter, test_iter

def text_to_img(text, TEXT):
    imgs = []
    for sample_ix in range(text.shape[1]):
        img_text = textwrap.wrap(" "*np.random.randint(0, 26) + " ".join(TEXT.vocab.itos[ix] for ix in text[:,sample_ix]), 27)
        img_text = "".join([ row + "\n" for row in img_text])

        img = Image.new('RGB', (224, 224), (0,0,0))
        draw = ImageDraw.Draw(img)
        draw.text((0,0), img_text, (255,255,255), font=font)
        draw = ImageDraw.Draw(img)

        imgs.append(normalize(img))
    return Variable(torch.stack(imgs).cuda())


