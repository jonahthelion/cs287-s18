import torchtext
from torchtext.vocab import Vectors
from tqdm import tqdm

from utils.models import TriGram
from utils.preprocess import get_data, get_model
from utils.postprocess import evaluate

# NOTE: success of TriGram will depend weakly on batch size
model_dict = {'max_size': 100, # max is 10001
                'batch_size': 10, 
                'bptt_len': 32,
                'num_epochs': 1,

                'type': 'trigram', 
                'alpha': [.1, .5, .4],}

train_iter, val_iter, test_iter, TEXT = get_data(model_dict)

model = get_model(model_dict)

for epoch in xrange(model_dict['num_epochs']):
    for batch in tqdm(train_iter):
        model.train()
        preds = model.train_predict(batch.text.cuda())
model.postprocess()

all_actuals, all_preds = evaluate(model, test_iter)

