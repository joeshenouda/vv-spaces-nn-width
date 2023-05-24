#!/bin/bash

cd ../
path_new_last='/home/jshenouda/vv-spaces-nn-width/results/alexnet/0411114323_mu_0.07_lam_0.05_last/V_new_iter_1400_act_797.npy'
path_new_pen='/home/jshenouda/vv-spaces-nn-width/results/alexnet/0411123755_mu_0.07_lam_0.5_pen/V_new_iter_1100_act_677.npy'
path_new_first='/home/jshenouda/vv-spaces-nn-width/results/alexnet/0411141424_mu_0.07_lam_0.5_first/V_new_iter_800_act_4896.npy'

python alexnet_test_compress.py --layer='pen' --path-new-last=$path_new_last --path-new-pen=$path_new_pen --path-new-first=$path_new_first