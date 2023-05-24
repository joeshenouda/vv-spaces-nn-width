import torch
import numpy as np
from models.alexnet import AlexNet
from datasets import CIFAR10_AlexNet
import argparse
from utils import eval_train, test
import os

parser = argparse.ArgumentParser(description='Train and Test on compressed AlexNet')
parser.add_argument('--layer', type=str, default='last', help='Layer to replace (last, pen, first or all)')
parser.add_argument('--path-new-last', type=str, default=None, help='Path to new last layer')
parser.add_argument('--path-new-pen', type=str, default=None, help='Path to new penultimate layer')
parser.add_argument('--path-new-first', type=str, default=None, help='Path to new first linear layer')
args = parser.parse_args()

device='cuda:0'

data_path = './data/cifar10'
samps=50000
cifar10_data = CIFAR10_AlexNet(data_path, samps=samps, bt_size=1024)

cifar10_train_loader = cifar10_data.train_loader
cifar10_test_loader = cifar10_data.test_loader

model = AlexNet(num_classes=10).to(device)
state_dict_path = os.path.join("./alexnet_pretrained/model_acc_72_wd.pt")
model.load_state_dict(torch.load(state_dict_path))

# Get original train and test accuracies
orig_test_acc = test(model, cifar10_test_loader, device=device)
orig_train_loss = eval_train(model, cifar10_train_loader, device=device)
print('Original Test Accuracy: {:.5f}'.format(orig_test_acc))
print('Original Train Loss: {:.5f}'.format(orig_train_loss), '\n')

# Replace linear layers with compressed versions
if args.path_new_last is not None:
    V_last = np.load(args.path_new_last)
if args.path_new_pen is not None:
    V_pen = np.load(args.path_new_pen)
if args.path_new_first is not None:
    V_first = np.load(args.path_new_first)

if args.layer == 'last':
    act_cols_last = 0
    for v_col in V_last.T:
        if np.linalg.norm(v_col) > 1e-7:
            act_cols_last += 1
    print('Active Neurons Last Layer {}'.format(act_cols_last))
    model.classifier[6].weight.data = torch.from_numpy(V_last).to(device)

elif args.layer == 'pen':
    act_cols_pen = 0
    for v_col in V_pen.T:
        if np.linalg.norm(v_col) > 1e-7:
            act_cols_pen += 1
    print('Active Neurons Penult Layer {}'.format(act_cols_pen))
    model.classifier[4].weight.data = torch.from_numpy(V_pen).to(device)

elif args.layer == 'first':
    act_cols_first = 0
    for v_col in V_first.T:
        if np.linalg.norm(v_col) > 1e-7:
            act_cols_first += 1
    print('Active Neurons First Linear Layer {}'.format(act_cols_first))
    model.classifier[1].weight.data = torch.from_numpy(V_first).to(device)

elif args.layer == 'all':
    act_cols_last = 0
    for v_col in V_last.T:
        if np.linalg.norm(v_col) > 1e-7:
            act_cols_last += 1
    print('Active Neurons Last Layer {}'.format(act_cols_last))
    model.classifier[6].weight.data = torch.from_numpy(V_last).to(device)

    act_cols_pen = 0
    for v_col in V_pen.T:
        if np.linalg.norm(v_col) > 1e-7:
            act_cols_pen += 1
    print('Active Neurons Penult Layer {}'.format(act_cols_pen))
    model.classifier[4].weight.data = torch.from_numpy(V_pen).to(device)

    act_cols_first = 0
    for v_col in V_first.T:
        if np.linalg.norm(v_col) > 1e-7:
            act_cols_first += 1
    print('Active Neurons Penult Layer {}'.format(act_cols_first))
    model.classifier[1].weight.data = torch.from_numpy(V_first).to(device)

# Get new train and test accuracies
new_test_acc = test(model, cifar10_test_loader)
new_train_loss = eval_train(model, cifar10_train_loader)
print('New Test Accuracy: {:.5f}'.format(new_test_acc))
print('New Train Loss: {:.5f}'.format(new_train_loss))
