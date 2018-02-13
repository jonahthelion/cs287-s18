import torchtext
from torchtext.vocab import Vectors
from tqdm import tqdm
import numpy as np

from utils.models import TriGram
from utils.preprocess import get_data, get_model
from utils.postprocess import evaluate, write_submission


# NOTE: success of TriGram will depend weakly on batch size
model_dict = {'max_size': 10001, # max is 10001
                'batch_size': 10, 
                'bptt_len': 32,
                'num_epochs': 1,

                'output': 'simple0.txt',

                # 'type': 'trigram', 
                # 'alpha': [.1, .5, .4],

                'type': 'NN',


                }

train_iter, val_iter, test_iter, TEXT = get_data(model_dict)

model = get_model(model_dict)

for epoch in range(model_dict['num_epochs']):
    for batch in tqdm(train_iter):
        model.train()
        preds = model.train_predict(batch.text.cuda())
        assert False
model.postprocess()


MAP = evaluate(model, val_iter)
print('MAP', MAP)
