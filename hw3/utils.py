import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_print(tensor):
    print("shape:", tensor.shape)
    print(tensor.cpu().numpy())

def review_print(text):
    print("#"*100)
    lst = text.split("<br />")
    for line in lst:
        if not line:
            print()
            continue
        else:
            l = len(line)
            n = l // 100 + 1
            for i in range(n):
                print(line[i*100:min((i+1)*100, l)])
    print("#"*100, end = "\n\n")

def normal(size):
    return torch.normal(mean = 0, std = 1, size = size)

def pos_or_neg(value):
   return "Positive" if value == 1 else "Negative"

def prediction_print(text, output, label):
    if len(text) > 80:
        print("- Raw text    :", text[:80], "...")
    else:
        print("- Raw text    :", text)
    print("- Model output:", output.squeeze(0).detach().cpu().numpy())
    print(f"- True label  : {pos_or_neg(label)} | Model prediction: {pos_or_neg(output.argmax().item())}\n")
    return True

def predict(model, text, label, tokenizer):
    x = torch.tensor(tokenizer(text), dtype = torch.int32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(x)
    prediction_print(text, output, label)
    
# Sentiment Analysis
def model_print(model, pos_text, neg_text, tokenizer):
    print(f"### Sentiment analysis of {model.__class__.__name__} model ###\n")
    right, wrong = False, False
    rand_idx = np.arange(len(pos_text))
    np.random.shuffle(rand_idx)
    for idx in rand_idx:
        text, label = pos_text[idx], 1
        x = torch.tensor(tokenizer(text), dtype = torch.int32).unsqueeze(0).to(DEVICE)
        output = model(x)
        if not right and output.argmax().item():
            print("True positive")
            right = prediction_print(text, output, label)
        if not wrong and not output.argmax().item():
            print("False negative")
            wrong = prediction_print(text, output, label)
        if right and wrong: break

    right, wrong = False, False
    for idx in rand_idx:
        text, label = neg_text[idx], 0
        x = torch.tensor(tokenizer(text), dtype = torch.int32).unsqueeze(0).to(DEVICE)
        output = model(x)
        if not right and not output.argmax().item():
            print("True negative")
            right = prediction_print(text, output, label)
        if not wrong and output.argmax().item():
            print("False positive")
            wrong = prediction_print(text, output, label)
        if right and wrong: break

def pos_enc_print(pos_enc):
    plt.figure(figsize = (25, 5))
    plt.imshow(pos_enc, cmap = "coolwarm_r")
    plt.colorbar(pad = 0.01)
    plt.xlabel("embedding dimension")
    plt.ylabel("position")
    plt.show()
