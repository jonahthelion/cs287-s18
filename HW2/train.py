import torchtext
from torchtext.vocab import Vectors
from tqdm import tqdm

from utils.models import TriGram
from utils.preprocess import get_data, get_model
from utils.postprocess import evaluate, write_submission

for alpha1 in [0.05,.1,.2,.3,.4,.5,.6,.7,.8,.9]:
    for alpha2 in np.linspace(0.05, .95-alpha1, 4):
        # NOTE: success of TriGram will depend weakly on batch size
        model_dict = {'max_size': 100, # max is 10001
                        'batch_size': 10, 
                        'bptt_len': 32,
                        'num_epochs': 1,

                        'output': 'simple0.txt',

                        'type': 'trigram', 
                        'alpha': [alpha1, alph2, 1-alpha1-alpha2],}

        train_iter, val_iter, test_iter, TEXT = get_data(model_dict)

        model = get_model(model_dict)

        for epoch in range(model_dict['num_epochs']):
            for batch in tqdm(train_iter):
                model.train()
                preds = model.train_predict(batch.text.cuda())
        model.postprocess()

        nll_l, MAP = evaluate(model, test_iter)
        print(model_dict['alpha'])
        print(nll_l, MAP)
        # samples, top_ranks = write_submission(model, model_dict['output'], TEXT)

