import torchtext
from torchtext.vocab import Vectors

max_size = 10002 # max is 10001
batch_size = 10
bptt_len = 32

TEXT = torchtext.data.Field()

train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="PSET", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

TEXT.build_vocab(train, max_size=max_size)

train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), batch_size=batch_size, device=-1, bptt_len=bptt_len, repeat=False)

for batch in train_iter:
    assert False