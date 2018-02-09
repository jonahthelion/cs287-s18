import torch

from tqdm import tqdm

def evaluate(model, test_iter):
    all_actuals = []
    all_preds = []
    for batch in tqdm(test_iter):
        text = torch.stack([ batch.text[:,i] for i in range(batch.text.shape[1]) if batch.text.data[-1, i] != 3]).t() 

        preds = model.predict(text.cuda()).cpu()

        labels = text[-1]

        all_actuals.extend(labels)
        all_preds.extend(preds)

    return all_actuals, all_preds


