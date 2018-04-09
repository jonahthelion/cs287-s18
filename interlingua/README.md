# Interlingua

1. Get the Multi30k data
`git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset`
`cd interlingua`

2. run a script which prints the first training batch
`python3 train.py -d ../multi30k-dataset -batch_size 4`

_Notes_

* batches returned by trainloader are dictionaries with keys ``'en','de','fr'`` and values translations of a random sentence from the Multi30k training set.
* tokenizing handled by ``spacy``
* `valloader` has identical format to `trainloader` but sampling is not random in `valloader`
* `TEXT.itos` and `TEXT.stoi` are dictionaries for the integer to (`lang`, `word`) mapping used by both trainloader and testloader.
* -2 is used to pad all sentences to length 15