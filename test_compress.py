import torch
import numpy as np
from models import LowRankVGG19, FullRankVGG19
from datasets import CIFAR10
import argparse
import os

parser = argparse.ArgumentParser(description='Train and Test on compressed VGG19')
parser.add_argument('--path-new-V', type=str, help='Path to new V')

args = parser.parse_args()

def test(model, test_loader):
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

def eval_train(model, train_loader):
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
            print(loss_fn(outputs, targets).item())
            train_loss += loss_fn(outputs, targets).item()
            
    return train_loss

device='cuda:0'

data_path = './data/cifar10'
samps=50000
cifar10_data = CIFAR10(data_path, samps=samps, bt_size=1024)

cifar10_train_loader = cifar10_data.train_loader
cifar10_test_loader = cifar10_data.test_loader

vgg_vanilla_model = torch.load('./pufferfish_checkpoint/vanilla_vgg19_seed2_best.pth')
vgg19_vanilla = FullRankVGG19().to(device)
vgg19_vanilla.load_state_dict(vgg_vanilla_model['net'])

# Get original train and test accuracies
orig_test_acc = test(vgg19_vanilla, cifar10_test_loader)
orig_train_loss = eval_train(vgg19_vanilla, cifar10_train_loader)
print('Original Test Accuracy: {:.5f}'.format(orig_test_acc))
print('Original Train Loss: {:.5f}'.format(orig_train_loss), '\n')

# Replace final linear with new V
V_new = np.load(args.path_new_V)
act_cols = 0
for v_col in V_new.T:
    if np.linalg.norm(v_col) > 1e-7:
        act_cols += 1
print('Active Neurons {}'.format(act_cols))
vgg19_vanilla.classifier[6].weight.data = torch.from_numpy(V_new).float().to(device)

import ipdb; ipdb.set_trace()
# Get new train and test accuracies
new_test_acc = test(vgg19_vanilla, cifar10_test_loader)
new_train_loss = eval_train(vgg19_vanilla, cifar10_train_loader)
print('New Test Accuracy: {:.5f}'.format(new_test_acc))
print('New Train Loss: {:.5f}'.format(new_train_loss))
