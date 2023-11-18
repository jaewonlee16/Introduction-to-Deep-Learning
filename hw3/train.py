import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
from torch import optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(epoch, model, optimizer, train_loader, use_mask, small_loader):
    model.train()
    train_loss = []

    if small_loader:
        iterator = train_loader 
    else:
        iterator = tqdm(train_loader,
                        total = len(train_loader),
                        desc = f"Epoch {epoch}",
                        leave = True)

    for batch in iterator:
        x, y, mask = batch
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()

        if use_mask:
            output = model(x, mask)
        else:
            output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        
    return train_loss

def evaluate(model, loader, use_mask):
    model.to(DEVICE)
    model.eval()
    num_data = len(loader.dataset)
    num_correct, total_loss = 0, 0

    for batch in loader:
        x, y, mask = batch
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

        if use_mask:
            output = model(x, mask)
        else:
            output = model(x)
        loss = F.cross_entropy(output, y, reduction = 'sum')
        total_loss += loss.item()

        num_correct += (output.max(dim = 1)[1].view(y.size()).data == y.data).sum().item()

    avg_loss = total_loss / num_data
    accuracy = num_correct / num_data
    return avg_loss, accuracy

def train(model, loaders, epochs, use_mask = False, lr = 1e-4, plot = True, small_loader = False):
    
    model.to(DEVICE)

    train_loader, val_loader, test_loader = loaders
    loss_log = []
    acc_log = []

    best_val_acc = None
    os.makedirs("./snapshot", exist_ok = True)

    optimizer = optim.Adam(model.parameters(), lr = lr)

    for epoch in range(1, epochs + 1):

        loss_log += train_epoch(epoch = epoch,
                                model = model,
                                optimizer = optimizer,
                                train_loader = train_loader,
                                use_mask = use_mask,
                                small_loader = small_loader)
        
        val_loss, val_acc = evaluate(model, val_loader, use_mask)
        acc_log.append(val_acc)
        if not small_loader:
            print(f"Validation loss: {val_loss:7.4f}| Validation accuracy: {val_acc:5.2f}")
        if best_val_acc is None or val_acc >= best_val_acc:
            torch.save(model.state_dict(), f"./snapshot/{model.__class__.__name__}.pt")
            best_val_acc = val_acc

    # Model loading
    model.load_state_dict(torch.load(f"./snapshot/{model.__class__.__name__}.pt"))
    test_loss, test_acc = evaluate(model, test_loader, use_mask)
    print(f"\nTest loss: {test_loss:7.4f}| Test accuracy: {test_acc:5.2f}\n")

    # Loss, accuracy plotting
    if plot:
        fig, ax = plt.subplots(1, 2, figsize = (8, 3))
        ax[0].plot(loss_log, color = "r")
        ax[0].set_xlabel("Iterations")
        ax[0].set_ylabel("Average CE loss")
        marker_size = 10 if small_loader else None
        ax[1].scatter(np.arange(1, epochs+1), acc_log, color = "g", marker = "s", s = marker_size)
        ax[1].plot(np.arange(1, epochs+1), acc_log, color = "g")
        step = max(epochs // 5, 1)
        ax[1].set_xticks(np.arange(0, epochs+1, step), np.arange(0, epochs+1, step))
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Validation accuracy")
        fig.supxlabel(f"{model.__class__.__name__} Training Results")
        plt.tight_layout()
        plt.show()




    