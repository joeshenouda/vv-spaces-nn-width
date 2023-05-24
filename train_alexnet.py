import os
import random
import argparse
import datetime

import numpy as np

import torch
import torch.nn as nn

from datasets.cifar10 import CIFAR10_AlexNet
from models.alexnet import AlexNet
from feature_extractor import FeatureExtractor
from utils import test
import wandb

parser = argparse.ArgumentParser(description='Train AlexNet on CIFAR10')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
parser.add_argument('--log-interval', type=int, default=100, help='Log interval')
parser.add_argument('--save-dir', type=str, default='./alexnet_pretrained', help='Directory to save model')
parser.add_argument('--wd', type=float, default=0.0001,help='Weight decay')
parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
args = parser.parse_args()


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)

#########################
######## WANDB ##########
#########################

if args.wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Train AlexNet on CIFAR-10",
        name = 'lr_{}_wd_{}_bs_{}'.format(args.lr, args.wd, args.batch_size),
        # track hyperparameters and run metadata
        config=args
    )
else:
    wandb=None


##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = args.lr
WD = args.wd
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

# Architecture
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"

set_all_seeds(RANDOM_SEED)

# Deterministic behavior not yet supported by AdaptiveAvgPool2d
#set_deterministic()


##########################
### CIFAR-10 DATASET ####
##########################

cifar10_alex = CIFAR10_AlexNet('./data/cifar10', bt_size=BATCH_SIZE)
train_loader = cifar10_alex.train_loader
test_loader = cifar10_alex.test_loader


torch.manual_seed(RANDOM_SEED)

model = AlexNet(NUM_CLASSES)
model.to(DEVICE)

print('Setting up feature extractor...')
model_w_feats = FeatureExtractor(model.eval(), ['classifier.1','classifier.4','classifier.6'])


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WD)  
loss_fn = nn.CrossEntropyLoss()

print('Start training...')
for epoch in range(NUM_EPOCHS):

    model.train()
    features = {layer: {'input': torch.empty(0), 'output': torch.empty(0)} for layer in model_w_feats.layers}
    for batch_idx, (imgs, targets) in enumerate(train_loader):

        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        # TWO FORWARD PASSES ONE FOR FEATURES AND ONE FOR LOSS
        output = model(imgs)

        _,feats = model_w_feats(imgs)

        for layer in model_w_feats.layers:
            #import ipdb; ipdb.set_trace()
            #if layer == 'classifier.1':
            #    features[layer] = features[layer].view(features[layer].size(0), 256 * 6 * 6)
            features[layer]['input'] = torch.cat((features[layer]['input'].detach().to('cpu'), feats[layer]['input'].detach().to('cpu')))
            features[layer]['output'] = torch.cat((features[layer]['output'].detach().to('cpu'), feats[layer]['output'].detach().to('cpu')))


        loss = loss_fn(output, targets)
        optimizer.zero_grad()

        loss.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                    % (epoch+1, NUM_EPOCHS, batch_idx,
                        len(train_loader), loss))

    # LOGGING
    wandb.log({'train_loss': loss.item()})


test_acc = test(model, test_loader)

# Save model
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
now = datetime.datetime.now().strftime('%m%d%H%M%S')


torch.save(model.state_dict(), os.path.join(args.save_dir, '{}_alexnet_model_acc_{}.pth'.format(now, test_acc)))