# Vector-Valued Variation Spaces and Bounds on Neural Network Width

Codebase for generating experimental results in the paper "Vector-Valued Variation Spaces and Bounds on Neural Network Width"

## Basic Setup
- Clone this repository
- Create a new ``conda`` environment using the provivded ``environment.yml`` file to ensure you are using the same dependencies.
``conda env create -f environment.yml``

## Overview of Experiments
Our paper consisted of two experiments. The first demonstrates our bound for the group lasso problem on synthetic data. The second takes a pre-trained VGG-19 architecture and compresses the neurons in the final fully connected layer.

## Compressing VGG-19 Experiments
First download the pre-trained model from [here](https://drive.google.com/file/d/1XdUH1vK3roVGKtu0UUng0pd5SLqfO6S_/view?usp=sharing), and put it in a new directory called ``vgg_pretrained``. Then run the executable script ``launch/vgg_compress_experiment.sh``. This will run for about 3 hours and save the results in a new directory called ``results/``. Run the script ``test_compress.py`` with an argument to the path of the new weights learned.

``python test_compress.py --path-new-V='./results/0222152359_mu_0.1_lam_0.0001_N_50000/V_new.npy` ``

This will print out the original train/test loss as well as the new train/test loss when replacing the weights of the last linear layer with the ones we just learned.

Experiments were run on NVIDIA GeForce RTX 3090 and took about 3-5 hours to finish compressing the network.

## Verifying Bounds for Group lasso  

