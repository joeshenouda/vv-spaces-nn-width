import numpy as np
import os
import argparse
import datetime
import torch
import wandb
from torch.utils.data import Dataset
from torchvision import datasets

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, mybias):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)
        
        # Fix bias
        self.linear.bias.data = mybias
        self.linear.bias.requires_grad=False
    
    def forward(self, x):
        out = self.linear(x)
        return out

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def train_mse_pytorch(X, Y, bias, dest_dir, tol=1e-6, lam=1e-2, mu=1e-6, pre_trained_V=None, wandb=None, device='cuda:1', save_interval=50000):
    k = X.shape[0]
    n = X.shape[1]
    d = Y.shape[0]

    ## Create linear model for Pytorch to optimize
    model = linearRegression(k, d, bias).to(device)

    X = X.to(device)
    Y = Y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=mu)
    loss_fn = torch.nn.MSELoss()

    act_cols_arr = []
    gl_arr = []
    mses_arr = []
    import ipdb; ipdb.set_trace()

    iters = 0
    keep_going=True
    while keep_going:
        gl_curr = 0
        active_curr = 0
        mse_curr = 0

        optimizer.zero_grad()
        ## Compute output on entire data
        output = model(X.T)

        ## Get the loss
        loss = loss_fn(output, Y.T)
        mse_curr = loss.item()

        ## Gradient step
        loss.backward()
        optimizer.step()
        
        ## Soft-thresholding on cols of V
        with torch.no_grad():
            V_k = model.linear.weight.data
            for i, v_col in enumerate(V_k.T):
                v_norm = torch.linalg.norm(v_col).item()
                gl_curr += v_norm
                eps = 1e-8
                # Soft-thresholding
                a = max(0, 1 - mu*lam/(v_norm+eps))

                V_k[:,i] = a * v_col
                if a > 0:
                    active_curr += 1
            
            model.linear.weight.data = V_k

        act_cols_arr.append(active_curr)
        gl_arr.append(gl_curr)
        mses_arr.append(mse_curr)

        if wandb:
            wandb.log({'active':active_curr,'gl':gl_curr,'mse':mse_curr, 'lam':lam, 'mu':mu})

        
        print('Iter: {} Active columns {}'.format(iters, active_curr))
        print('MSE Original Output: {}'.format(mse_curr))
        print('Group LASSO Loss: {}'.format(gl_curr),'\n')
        
        if (iters + 1)%1000 == 0:
            keep_going = stopping_criterion(mse_curr, mse_prev, gl_curr, gl_prev, tol=tol)
        
        if iters % save_interval == 0:
            print('Saving current checkpoint')
            np.save(os.path.join(dest_dir, 'V_new_iter_{}_act_{}.npy'.format(iters,active_curr)), V_k.detach().cpu().numpy())
        iters += 1
        gl_prev = gl_curr
        mse_prev = mse_curr
    
    V_new = model.linear.weight.data
    return V_new, act_cols_arr, mses_arr,gl_arr

def stopping_criterion(mse_curr, mse_prev, gl_curr, gl_prev, tol=1e-5):
    rel_delta_GL = np.abs(gl_curr - gl_prev) / gl_prev
    rel_delta_MSE = np.abs(mse_prev - mse_curr) / mse_prev
    
    print('Rel. Delta GL {}'.format(rel_delta_GL))
    print('Rel. Delta Loss {}'.format(rel_delta_MSE))

    if rel_delta_GL < tol and rel_delta_MSE < tol:
        print('Barely changed, stopping')
        go = False
    else:
        go = True
    
    return go
