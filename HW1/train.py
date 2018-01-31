import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable

train_iter, val_iter, test_iter = get_data(batch_size=10)

print (len(train_iter))
print (len(val_iter))
print (len(test_iter))
