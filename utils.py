import torch
import numpy as np
import torch.nn as nn

def test(model, test_loader, device='cuda:0'):
    '''
    Evaluate the model on the test set and return accuracy
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.*correct/total

def eval_train(model, train_loader, device='cuda:0'):
    '''
    Evaluate the model on the training set
    '''
    model.eval()
    train_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.manual_seed(43)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            train_loss += loss_fn(outputs, targets).item()
    train_loss /= len(train_loader)
    return train_loss

def test_loss(model, test_loader, loss_fcn, device='cuda:0'):
    '''
    Evaluate the model on the test set and return accuracy
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loss = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += loss_fcn(outputs, targets).item()
    return loss