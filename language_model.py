import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import PIL
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import tqdm
import torchmetrics as tm
import torchtext
from torchtext.datasets import WikiText2
from torch.utils.data.dataset import IterableDataset
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

## Dataset

class TextFileDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                processed_line = line.strip()
                yield processed_line

# file_path = 'large_text_file.txt'
# dataset = TextFileDataset(file_path)

# train_iter, test_iter, valid_iter = WikiText2("./")
# next(iter(train_iter))

dataset = load_dataset(path="wikitext", name="wikitext-2-v1") # , split="train"
train_iter, test_iter, valid_iter = iter(dataset['train']), iter(dataset['test']), iter(dataset['validation'])

# read sample of train dataset
for i in range(3):
    print(next(train_iter)['text'])
## example

tokenizer = get_tokenizer('basic_english')
# txt = "Hi Ebrahim Parcham! 1 2 324 n11 #45"
# print(tokenizer(txt))

txt = ["hi there", "how are you"]
# print(list(map(tokenizer, txt))) ## map can use fun with iter list

vocab = build_vocab_from_iterator(map(tokenizer, txt), specials = ['<ukn>'], min_freq = 1) # use '<ukn>' to uknow data index for first and min_freq is min freq use in dict
print(vocab.get_stoi())
vocab.set_default_index(vocab['<ukn>'])  # set ukn to unknow vocabulatry
## infer index

print(vocab["hi"])

## use train iter
tokenizer = get_tokenizer('basic_english')
texts = (example['text'] for example in iter(dataset['train']))
vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=['<ukn>'])
vocab.set_default_index(vocab['<ukn>'])
# print(vocab.get_stoi())

# EDA

# Ensure 'text' is available in your dataset
# print(dataset['train'][0]['text'])  # Check if text data is present in the first example

texts = (example['text'] for example in iter(dataset['train']))
freq = Counter()
for token in map(tokenizer, texts):
    freq.update(token)
    # print(token)  # Debugging: Print out tokens to see if anything is being yielded

print(freq.most_common())