# Vector-Valued Variation Spaces and Width Bounds for Deep Neural Networks: Insights on Weight Decay Regularization

Codebase for reproducing experimental results in the paper "Vector-Valued Variation Spaces and Width Bounds for Deep Neural Networks: Insights on Weight Decay Regularization"-Joseph Shenouda, Rahul Parhi, Kangwook Lee and Robert D. Nowak

## Basic Setup
- Clone this repository
- Create a new ``conda`` environment using the provivded ``environment.yml`` file to ensure you are using the same dependencies.
``conda env create -f environment.yml``

## Overview of Experiments
Our paper consisted of three sets of experiments. The first illustrates the neuron sharing phenomenon on a synthetic dataset. The second set of experiments validate theoretical bound for the sparsity in solutions to the multi-task lasso problem on synthetic data. The third set of experiments show how we can use this multi-task lasso problem to compress a pre-trained deep neural networks, specifically VGG-19 and AlexNet.

## Neuron Sharing Experiments
All the code for generating the figures for this experiments can be found in the Jupyter notebook ``neuron_sharing_exp.ipynb`` this can be run locally on a CPU.

## Multi-Task Lasso Experiments
All the code for reproducing the results on the first set of experiments (Fig. 3 and 4) can be found in ``cvx_multi_task_lasso.ipynb``. 

The code to reproduce Fig. 5 can be found in ``brute_cvx_multi_task_lasso.py`` to reproduce each subplot run 

``python brute_cvx_multi_task_lasso.py --N=3 --D=2 --K=7``

for K=7,8,9,10,11. The histograms were generated according to the code in ``plotting_cvx_experiments.ipynb``.
Both of these experiments only used the CPU.
## Compressing Deep Neural Networks
### Compressing VGG-19
First download the pre-trained model from [here](https://drive.google.com/file/d/1XdUH1vK3roVGKtu0UUng0pd5SLqfO6S_/view?usp=sharing), and put it in a new directory called ``vgg_pretrained``. To compress the last linear layer of the model run the script ``vgg19_compress_final_layer.py`` with mu=0.03 and lam=0.01. We stopped running once we saw convergence. 

 Run the script ``vgg_test_compress.py`` with an argument to the path of the new weights learned.

For example,

``python test_compress.py --path-new-V='./results/0222152359_mu_0.1_lam_0.0001_N_50000/V_new.npy` ``

This will print out the original train/test loss as well as the new train/test loss when replacing the weights of the last linear layer with the ones we just learned.

## Compressing AlexNet
We trained our own AlexNet first using the script ``train_alexnet.py``. Then run ``alexnet_compress.py`` with a different argument for ``--layer`` depending on the layer you want to compress (last, pen or first) corresponding to the last ReLU layer, the penultimate ReLU layer and the first ReLU layer. 
To compress the last linear layer we use arguments mu=0.07 and lam=0.05, for the penultimate layer mu=0.07 lam=0.5 and similarly for the first linear layer. We manually stopped running proximal gradient descent once we saw convergence.
 
Experiments were run on NVIDIA GeForce RTX 3090 and took about 3-5 hours to finish compressing the network.


