import torch
import numpy as np
from models import FullRankVGG19
from datasets import CIFAR10_DA
from utils import eval_train, test
import argparse
import os

parser = argparse.ArgumentParser(description='Train and Test on compressed VGG19')
parser.add_argument('--path-new-V_last', type=str, help='Path to new V For Last Layer')

args = parser.parse_args()


device='cuda:0'

data_path = './data/cifar10'
samps=50000
cifar10_data = CIFAR10_DA(data_path, samps=samps, bt_size=1024)

cifar10_train_loader = cifar10_data.train_loader
cifar10_test_loader = cifar10_data.test_loader

vgg_vanilla_model = torch.load('./vgg_pretrained/vanilla_vgg19_seed2_best.pth')
vgg19_vanilla = FullRankVGG19().to(device)
vgg19_vanilla.load_state_dict(vgg_vanilla_model['net'])

# Get original train and test accuracies
orig_test_acc = test(vgg19_vanilla, cifar10_test_loader, device=device)
orig_train_loss = eval_train(vgg19_vanilla, cifar10_train_loader, device=device)
print('Original Test Accuracy: {:.5f}'.format(orig_test_acc))
print('Original Train Loss: {:.5f}'.format(orig_train_loss), '\n')

vgg19_compressed = FullRankVGG19().to(device)
vgg19_compressed.load_state_dict(vgg_vanilla_model['net'])

# Replace final linear with new V
V_new_last = np.load(args.path_new_V_last)
act_cols_last = 0
for v_col in V_new_last.T:
    if np.linalg.norm(v_col) > 1e-7:
        act_cols_last += 1
print('Active Neurons Last Layer {}'.format(act_cols_last))

vgg19_compressed.classifier[6].weight.data = torch.from_numpy(V_new_last).float().to(device)

# Get new train and test accuracies
new_test_acc = test(vgg19_compressed, cifar10_test_loader)
new_train_loss = eval_train(vgg19_compressed, cifar10_train_loader)
print('New Test Accuracy: {:.5f}'.format(new_test_acc))
print('New Train Loss: {:.5f}'.format(new_train_loss))
