with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

    print("number of chars in dataset:", len(text))
    print("first 1000 chars", text[:1000])

    # all unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
 
# character level language model
# mapping chars to numbers

    # encoder: take a string output integers
    # This dictionary assigns each character a unique index
stoi = { ch:i for i,ch in enumerate(chars) }
    # Encodes the given string with dictionary
encode = lambda s: [stoi[c] for c in s] 
    
    # decoder: take in list of integers, produce string
itos = { i:ch for i,ch in enumerate(chars)}
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('hi there'))
print(decode(encode('hi there')))

import torch
# creates a tensor (essentially a matrix or array) from some encoded version of the text
# ensures that each element in the tensor is treated as a long integer (useful when handling tokenized text for models)
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# split up the data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, the target: {target}")
    
# generate random starting points within the range of len(data) - block_size
torch.manual_seed(1337)
batch_size = 4 # Sequences processed in parallel
block_size = 8 # Max content length

#  function picks either train_data or val_data
def get_batch(split):
    # generate batch of inputs (x) and targets (y)
    data = train_data if split == 'train' else val_data
    
    # stack those chunks into tensors! Specifically, for each starting index in ix
    # grab a slice of length block_size for x and the next slice for y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f'when input is {context.tolist()} the target {target}')

print(xb) # input to transformer

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

