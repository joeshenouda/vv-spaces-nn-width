from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn
from models import ShallowNet

class synthetic_dataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class synthetic():
    def __init__(self, x_train, y_train):
        train_dataset = synthetic_dataset(x_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=x_train.shape[0], num_workers=4)

# Generates a synthetic dataset where the ground truth is itself a neural network
class nn_dataset():
    def __init__(self, input_dim, out_dim, samps, rand_seed=43, correlated=False):
        torch.manual_seed(rand_seed)
        x_train = torch.randn(samps, input_dim)
        x_val = torch.randn(samps//2, input_dim)
        x_test = torch.randn(samps//2, input_dim)

        if correlated:
            net_gen = ShallowNet(input_dim, width = 1, num_classes=out_dim)
            y_train = net_gen(x_train).detach() + 0.05*torch.randn(samps, out_dim)
            y_val = net_gen(x_val).detach()
            y_test = net_gen(x_test).detach()

            self.train_loader = torch.utils.data.DataLoader(synthetic_dataset(x_train, y_train), batch_size=samps, num_workers=4)
            self.val_loader = torch.utils.data.DataLoader(synthetic_dataset(x_val, y_val), batch_size=x_val.shape[0], num_workers=4)
            self.test_loader = torch.utils.data.DataLoader(synthetic_dataset(x_test, y_test), batch_size=x_test.shape[0], num_workers=4)
        else:
            # make the output weights of each neuron 1 sparse such that each output of the network is only dependent on one neuron
            net_gen = ShallowNet(input_dim, width = out_dim, num_classes=out_dim)
            net_gen.fc2.weight.data = torch.eye(out_dim)

            y_train = net_gen(x_train).detach() + 0.05*torch.randn(samps, out_dim)
            y_val = net_gen(x_val).detach()
            y_test = net_gen(x_test).detach()
            self.train_loader = torch.utils.data.DataLoader(synthetic_dataset(x_train, y_train), batch_size=len(x_train), num_workers=4)
            self.val_loader = torch.utils.data.DataLoader(synthetic_dataset(x_val, y_val), batch_size=len(x_val), num_workers=4)
            self.test_loader = torch.utils.data.DataLoader(synthetic_dataset(x_test, y_test), batch_size=len(x_test), num_workers=4)



