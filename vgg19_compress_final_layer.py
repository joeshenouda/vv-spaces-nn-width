import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import FullRankVGG19
from datasets import CIFAR10_DA
import argparse
import os
import datetime
from prox_gradient import train_mse_pytorch


import wandb

parser = argparse.ArgumentParser(description='Compress VGG-19 Pretrained on CIFAR-10 using proximal gradient')
parser.add_argument('--tol', type=float, default=1e-7, help='Tolerance for stopping')
parser.add_argument('--lam', type=float, default=1e-2, help='Group lasso Regularization')
parser.add_argument('--mu', type=float, default=3e-3, help='Step size for GD')
parser.add_argument('--wandb', type=int, default=0, help='User Wandb')

args = parser.parse_args()


model_name = 'vgg19_vanilla'

if args.wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Compress VGG19",
        name = 'mu_{}_lam_{}_tol_{}'.format(args.mu, args.lam, args.tol),
        # track hyperparameters and run metadata
        config={
        "mu": args.mu,
        "lam": args.lam,
        "architecture": "VGG19",
        "dataset": "CIFAR-10",
        "tol": args.tol,
        }
    )
else:
    wandb=None
device='cuda:1'

data_path = './data/cifar10'
samps=50000
cifar10_data = CIFAR10_DA(data_path, samps=50000, bt_size=2048)

cifar10_train_loader = cifar10_data.train_loader
cifar10_test_loader = cifar10_data.test_loader


vgg_vanilla_model = torch.load('./vgg_pretrained/vanilla_vgg19_seed2_best.pth')
vgg19_vanilla = FullRankVGG19()
vgg19_vanilla.load_state_dict(vgg_vanilla_model['net'])

vgg19_vanilla.to(device)

# Use hooks to get features passed into final layer
class features:
    pass

def hook(self, input, output):
    features.value = input[0].clone()

classifier = vgg19_vanilla.classifier._modules['6'] # final layer of VGG-19 model
classifier.register_forward_hook(hook)

torch.manual_seed(43)

penult_feat_batches = []
output_vgg_no_bias_batches = []
output_vgg_batches = []
cifar10_data_labels = []

train_loss=0
loss_fn = torch.nn.CrossEntropyLoss()

## Get penultimate representation of entire dataset into a single PyTorch Tensor
with torch.no_grad():
    for data, target in cifar10_train_loader:
        data, target = data.to(device), target.to(device)
        output = vgg19_vanilla(data)

        # Extract representation entering penultimate layer
        h = features.value.data.view(data.shape[0],-1)

        output_no_bias = output - vgg19_vanilla.classifier._modules['6'].bias
        output_vgg_no_bias_batches.append(output_no_bias)
        output_vgg_batches.append(output)

        penult_feat_batches.append(h)
        cifar10_data_labels.append(target.detach())

        train_loss += loss_fn(output, target).item()  # sum up batch loss

penult_feats = torch.cat(tuple(penult_feat_batches), 0)  # 512 X 50K
output_vgg_no_bias = torch.cat(tuple(output_vgg_no_bias_batches)) # 10 X 50K
output_vgg = torch.cat(tuple(output_vgg_batches))

cifar10_data_labels = torch.cat(tuple(cifar10_data_labels),0) # 1 X 50K


Phi = penult_feats.data.T
Psi = output_vgg.data.T
V_orig = vgg19_vanilla.classifier._modules['6'].weight.data
V_np = V_orig.detach().cpu().numpy().T
d = Psi.shape[0]
n = Phi.shape[1]
k = Phi.shape[0]

mu=args.mu
lam=args.lam


now = datetime.datetime.now().strftime('%m%d%H%M%S')
dest_dir = 'results/{}_mu_{}_lam_{}_N_{}'.format(now, args.mu, args.lam, 50000)
os.makedirs(dest_dir, exist_ok=True)


np.save(os.path.join(dest_dir, 'X.npy'), Phi.cpu().numpy())
np.save(os.path.join(dest_dir, 'Y.npy'), Psi.cpu().numpy())


bias = vgg19_vanilla.classifier._modules['6'].bias.data

V_new, actives_new, mses_new, GL_new = train_mse_pytorch(Phi, Psi, bias, dest_dir, tol=args.tol, lam=lam, mu=mu, pre_trained_V=None, wandb=wandb)
np.save(os.path.join(dest_dir, 'V_new.npy'), V_new.cpu().numpy())
np.save(os.path.join(dest_dir, 'actives_new.npy'), actives_new)
np.save(os.path.join(dest_dir, 'mses_new.npy'), mses_new)
np.save(os.path.join(dest_dir, 'GL_new.npy'), GL_new)


