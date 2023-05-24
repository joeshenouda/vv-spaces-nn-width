import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
import os
import argparse
import datetime


from models.alexnet import AlexNet
from datasets import CIFAR10_DA, CIFAR10_AlexNet


from prox_gradient import train_mse_pytorch
import wandb

parser = argparse.ArgumentParser(description='Compress Pretrained ALexNet on CIFAR-10 using proximal gradient')
parser.add_argument('--tol', type=float, default=1e-7, help='Tolerance for stopping')
parser.add_argument('--lam', type=float, default=1e-2, help='Group lasso Regularization')
parser.add_argument('--mu', type=float, default=3e-3, help='Step size for GD')
parser.add_argument('--layer', type=str, default='last', help='Linear layer to compress (last,pen or first)')
parser.add_argument('--wandb', type=int, default=0, help='User Wandb')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0 or cuda:1)')

args = parser.parse_args()

model = AlexNet(num_classes=10)
state_dict_path = os.path.join(
                "./alexnet_pretrained/model_acc_72_wd.pt"
            )
model.load_state_dict(torch.load(state_dict_path))

# Get CIFAR-10 data
data_path = './data/cifar10'
samps=50000
cifar10_data = CIFAR10_AlexNet(data_path, samps=50000, bt_size=512)

cifar10_train_loader = cifar10_data.train_loader
cifar10_test_loader = cifar10_data.test_loader

from feature_extractor import FeatureExtractor
device = args.device if torch.cuda.is_available() else 'cpu'
#device='cpu'

# Extract features coming into a particular FFn layer
if args.layer == 'last':
    layer_name = 'classifier.6'
elif args.layer == 'pen':
    layer_name = 'classifier.4'
elif args.layer == 'first':
    layer_name = 'classifier.1'

model_feats = FeatureExtractor(model.to(device).eval(), [layer_name])


def get_features(model_feats, loader, device):
    '''
    Get features of entire dataset from a model Feature Extractor
    '''
    with torch.no_grad():
        features = {layer: {'input': torch.empty(0), 'output': torch.empty(0)} for layer in model_feats.layers}
        for imgs, label in loader:
            imgs = imgs.to(device)
            feats = model_feats(imgs)
            for layer in model_feats.layers:
                features[layer]['input'] = torch.cat((features[layer]['input'].to('cpu'), feats[layer]['input'].to('cpu')))
                features[layer]['output'] = torch.cat((features[layer]['output'].to('cpu'), feats[layer]['output'].to('cpu')))
    return features

feats = get_features(model_feats, cifar10_train_loader, device)

Psi = feats[layer_name]['output'].T
Phi = feats[layer_name]['input'].T


bias = dict([*model.named_modules()])[layer_name].bias.detach().cpu()

dest_dir = './alexnet_compressed_{}'.format(args.layer)

if args.wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Compress AlexNet",
        name = 'mu_{}_lam_{}_tol_{}_layer_{}'.format(args.mu, args.lam, args.tol, args.layer),
        # track hyperparameters and run metadata
        config={
        "mu": args.mu,
        "lam": args.lam,
        "architecture": "AlexNet",
        "dataset": "CIFAR-10",
        "tol": args.tol,
        }
    )
else:
    wandb=None

mu=args.mu
lam=args.lam


now = datetime.datetime.now().strftime('%m%d%H%M%S')
dest_dir = 'results/alexnet/{}_mu_{}_lam_{}_{}'.format(now, args.mu, args.lam, args.layer)
os.makedirs(dest_dir, exist_ok=True)

np.save(os.path.join(dest_dir, 'X.npy'), Phi.cpu().numpy())
np.save(os.path.join(dest_dir, 'Y.npy'), Psi.cpu().numpy())


V_new, actives_new, mses_new, GL_new = train_mse_pytorch(Phi, Psi, bias, dest_dir, tol=1e-7, lam=lam, mu=mu, pre_trained_V=None, wandb=wandb, save_interval=100)
np.save(os.path.join(dest_dir, 'V_new.npy'), V_new.cpu().numpy())
np.save(os.path.join(dest_dir, 'actives_new.npy'), actives_new)
np.save(os.path.join(dest_dir, 'mses_new.npy'), mses_new)
np.save(os.path.join(dest_dir, 'GL_new.npy'), GL_new)
