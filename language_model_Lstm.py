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
# vocab.append_token(specials=['<eos>'])
# print(vocab.get_stoi())

# EDA

# Ensure 'text' is available in your dataset
# print(dataset['train'][0]['text'])  # Check if text data is present in the first example

texts = (example['text'] for example in iter(dataset['train']))
freq = Counter()
for token in map(tokenizer, texts):
    freq.update(token)
    # print(token)  # Debugging: Print out tokens to see if anything is being yielded

# print(freq.most_common())

def pross_data(row_data, seq_len = 120):
    a = [torch.LongTensor(vocab(tokenizer(line['text'])) + vocab(['<eos>'])) for line in row_data] ## list of tensor from vocab
    data = torch.cat(a) ## to cancat all data in one line
    M = len(data) // seq_len
    r = len(data) % seq_len
    data = torch.cat((data, torch.LongTensor([0]))) if r==0 else data  # sheft 1 vocab
    inputs = data[:M*seq_len]
    inputs = inputs.reshape(-1, seq_len)

    targets = data[1: M * seq_len + 1]
    targets = targets.reshape(-1, seq_len)
    return inputs, targets

x_train, y_train = pross_data(train_iter)
x_valid, y_valid = pross_data(valid_iter)
x_test, y_test   = pross_data(test_iter)

class language_model_dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]


train_set = language_model_dataset(x_train, y_train)
valid_set = language_model_dataset(x_valid, y_valid)
test_set  = language_model_dataset(x_test, y_test)


batch_size = 20

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size = batch_size, shuffle=False)

## MODEL

class language_model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 dropout_embd = 0.5, dropout_rnn = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1) ## random weight for embedding layares

        self.dropout = nn.Dropout(p=dropout_embd)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            num_layers = self.num_layers,
                            dropout = dropout_rnn, batch_first = True)

        self.fc = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, src):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding)
        pred = self.fc(output)

        return pred

model = language_model(vocab_size=len(vocab),
                       embedding_dim=300,
                       num_layers=2,
                       hidden_dim=512,
                       dropout_rnn=0.2,
                       dropout_embd=0.5)

def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

device = 'cuda' if torch.cuda.is_available() else 'cpu'

do_again = False
train_run = True
load_model = True
clip_size = 0.25

if load_model:
    model_path = 'model.pt'
    model = torch.load(model_path).to(device)
else:
    model = model.to(device)

print(f"num trainable params: {num_trainable_params(model)}")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4)
metric = tm.text.Perplexity().to(device)

def train_one_epoch(train_loader, model, loss_fn, device, optimizer, metric, epoch=None):
    print("Start train")
    model = model.train()
    loss_train = AverageMeter()
    metric.reset()
    with tqdm.tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch:
                tepoch.set_description(f"Epoch {epoch}")
            targets = targets.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
            loss.backward()

            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm = clip_size)


            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item(), n=len(targets))
            # outputs = (outputs > 0.5).int()
            metric.update(outputs, targets)
            tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())

    return model, loss_train.avg, metric.compute().item()

def evaluate(test_loader, model, loss_fn, device, metric):
    print("Start evaluate")
    model = model.eval()
    loss_test = AverageMeter()
    metric.reset()
    with torch.inference_mode():
        for inputs, targets in test_loader:
            targets = targets.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

            loss_test.update(loss.item(), n=len(targets))
            metric.update(outputs, targets)

    return  loss_test.avg, metric.compute().item()

if do_again:
    train_one_epoch(train_loader, model, loss_fn, device, optimizer, metric, epoch=1)
    evaluate(test_loader, model, loss_fn, device, metric)

if train_run:
    loss_train_hist = []
    loss_valid_hist = []

    metric_train_hist = []
    metric_valid_hist = []

    best_loss_valid = torch.inf
    epoch_counter = 0

    num_epochs = 100

    for epoch in range(num_epochs):
        # Train
        model, loss_train, metric_train = train_one_epoch(train_loader, model, loss_fn, device, optimizer, metric, epoch=epoch)
        # Validation
        loss_valid, metric_valid = evaluate(valid_loader, model, loss_fn, device, metric)

        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)

        metric_train_hist.append(metric_train)
        metric_valid_hist.append(metric_valid)

        if loss_valid < best_loss_valid:
            torch.save(model, f'model.pt')
            best_loss_valid = loss_valid
            print('Model Saved!')

        print(f'Valid: Loss = {loss_valid:.4}, Perplexity = {metric_valid:.4}')
        print()

        epoch_counter += 1


    plt.figure(figsize=(8, 6))

    plt.plot(range(epoch_counter), loss_train_hist, 'r-', label='Train')
    plt.plot(range(epoch_counter), loss_valid_hist, 'b-', label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    loss_test,metric_test = evaluate(test_loader, model, loss_fn, device, metric)
    print(f"loss test: {loss_test}")
else:
    def generate_text(model, vocab, tokenizer, prompt="Ebrhaim parcham ", max_seq_len=5, device='cuda'):
        itos = vocab.get_itos()
        indices = vocab(tokenizer(prompt))

        for i in range(max_seq_len):
            src = torch.LongTensor(indices).to(device)

            with torch.no_grad():
                pred = model(src)
            #      ## وقتی به سمت صفر حرکت میکنند به سمت اعداد بزرگتر حرکت میکنند  یعنی رفتار انتخاب بزرگترین را انتخاب میکنم ولی اگر به سمت یک برود که همه اعداد شانس یکسانی خواهند داشت
            temperature = 0.5
            prob = torch.softmax(pred[-1] / temperature, dim=0)

            idx = vocab['<ukn>']
            while idx in [vocab['<ukn>'], vocab['<unk>']]:
                idx = torch.multinomial(prob, num_samples=1).item()

            prompt = prompt + " " + itos[idx]
            indices.append(idx)
            if itos[idx] == ".":
                break

        return prompt

    prompt = "The sun is shining."
    # prompt = "I love to eat pizza."
    # prompt = "Learning new things is exciting."
    # prompt = "In a galaxy far, far away..."
    # prompt = "Once upon a time, there was a little mouse."
    # prompt = "The sound of waves crashing against the shore."
    # prompt = "Coding is like solving puzzles."
    # prompt = "A cup of tea on a rainy day."
    # prompt = "Walking through a forest in autumn."
    # prompt = "Stars twinkling in the night sky."
    generated_text = generate_text(model, vocab, tokenizer, prompt=prompt, max_seq_len=30)
    print(generated_text)
