import torchtext
from torchtext.vocab import Vectors

TEXT = torchtext.data.Field()

train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path="PSET", 
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
