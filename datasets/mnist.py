from torchvision import datasets, transforms
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import os
import numpy as np


def get_mnist(data_path, batch_size):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/',
                                train=True,
                                transform=transform,
                                download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                train=False,
                                transform=transform)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=10000,
                                            shuffle=False, num_workers=4)
    return train_loader, test_loader

def get_mnist_subset(data_path, label_include, num_samps):

    # Data loading code
    kwargs = {"num_workers": 4, "pin_memory": True}
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(data_path, train=True, 
                               download=True, transform=transform)

    # permute
    train_data = train_set.__dict__['data']
    train_targets = train_set.__dict__['targets']
    idx_rand = torch.randperm(60000, generator=torch.Generator().manual_seed(42))
    train_data = train_data[idx_rand]
    train_targets = train_targets[idx_rand]
    train_set.__dict__['data'] = train_data
    train_set.__dict__['targets'] = train_targets

    updated_train_data, updated_train_targets = [], []
    count = torch.zeros(label_include)
    for i in range(60000):
        target_i = train_set.__dict__['targets'][i].item()
        sample_i = train_set.__dict__['data'][i]
        if target_i in torch.arange(label_include).tolist() and count[target_i] < num_samps//label_include:
            updated_train_data.append(sample_i)
            updated_train_targets.append(target_i)
            count[target_i] += 1

    train_set.__dict__['targets'] = updated_train_targets
    train_set.__dict__['data'] = updated_train_data

    test_set = datasets.MNIST(data_path, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=len(train_set), shuffle=True, drop_last=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=len(test_set), shuffle=False, drop_last=False, **kwargs)

    return train_loader, test_loader