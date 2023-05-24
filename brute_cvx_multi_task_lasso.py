import numpy as np
np.set_printoptions(precision=5)
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from itertools import combinations
import cvxpy as cp
import argparse
import datetime
import time
from scipy.stats import ortho_group  # Requires version 0.18 of scipy

parser = argparse.ArgumentParser(description='Brute Force Multi-Task LASSO w. CVX')
parser.add_argument('--K',  type=int, help='number of columns in V matrix')
parser.add_argument('--N', type=int, help='number of training samples')
parser.add_argument('--D', type=int, default=2, help='Output dimension for data')
parser.add_argument('--iters', type=int, default=1000, help='Number of iterations to run for')
parser.add_argument('--rX', type=int, default=0, help='Explicitly set rank of X')
parser.add_argument('--rY', type=int, default=0, help='Explicitly set rank of Y')


args = parser.parse_args()

d = args.D
n = args.N
k = args.K
num_act = args.num_act
iters = args.iters
all_patterns = 1
rY = args.rY
if rY == 0:
    rY = d
rX = args.rX
if rX == 0:
    rX = n

def group_lasso_cvx(w,k,d,sparsity_pattern):
    sum = 0
    for i in range(k):
        include = sparsity_pattern[i]
        group_w = w[i*d:(i+1)*d]
        norm = cp.norm2(group_w)
        sum += include * norm
    return sum

def calc_group_lasso(V):
    v_k_norms_arr = []

    active = 0
    for v_k in V.T:
        v_k_norm = np.linalg.norm(v_k)
        v_k_norms_arr.append(v_k_norm)
    GL = np.sum(v_k_norms_arr)
    active = np.sum(np.asarray(v_k_norms_arr) > 1e-7)
    
    return GL, active

def cvx_group_lasso(X, Y,sparsity):    
    #Vectorize
    Z = np.kron(X.T, np.eye(d,d))
    Y_vec = Y.flatten('F')
    
    # Setup convex problem
    w = cp.Variable(d*k)
    data_fitting_obj = group_lasso_cvx(w,k,d, sparsity)
    objective = cp.Minimize(data_fitting_obj)
    constraint = [Z @ w == Y_vec]
    prob = cp.Problem(objective, constraint)

    # Solve

    result = prob.solve()
    w_star = np.asarray(w.value)
    V_star = w_star.reshape((d,k), order='F')
    
    GL_star, active_star = calc_group_lasso(V_star)
    
    mse_loss = np.linalg.norm(V_star @ X - Y)**2
    
    return GL_star, active_star, mse_loss, V_star, w_star

def brute_cvx_lasso(X, Y, rY):
    cnt = 0
    group_lassos = []
    errors = []

    actives = []
    Vs = []
    found=False

    # Iterate through all possible sparsity patterns with num_act
    for num_act in range(n,k+1):
        combos_num_act = list(combinations(range(k), num_act))
        for combo in combos_num_act:
            sparsity = np.zeros(k)
            combo = list(combo)
            sparsity[combo] = 1


            cnt += 1
            # Zero out columns of X.T by elements of v
            X_tilde = X.T * sparsity
            tic = time.perf_counter()
            GL, num_active, mse, V, w = cvx_group_lasso(X_tilde.T, Y, sparsity)
            toc = time.perf_counter()
            mse_org = np.linalg.norm(V @ X - Y)**2
            actives.append(num_active)
            group_lassos.append(GL)
            Vs.append(V)
            

    group_lassos = np.asarray(group_lassos)
    actives = np.asarray(actives)
    Vs = np.asarray(Vs)
    eps = 1e-6
    min_gl = np.min(group_lassos)
    arg_min_ish = np.argwhere(np.abs(group_lassos - min_gl) <= eps)
    actives_min_gl = actives[arg_min_ish]
    print('Min GL:{}'.format(min_gl))
    print('Active cols for min gl: {}'.format(np.min(actives_min_gl)))

    sorted_gl = np.sort(group_lassos)
    arg_sorted_gl = np.argsort(group_lassos)

    actives_gl_sorted =  actives[arg_sorted_gl]
    print('Sorted GL\n',sorted_gl)

    print('Corresponding actives\n', actives_gl_sorted)

    return np.min(actives_min_gl)


act_cols_arr = []
good_XY = []
for iter in tqdm(range(iters)):
    print('Iteration: {}'.format(iter))
    X = np.random.randn(k,n)
    Y = np.random.randn(d,n)

    if rY < d:
        U = np.random.randn(rY, n) # rank rY, the n columns are linearly independent w.h.p.

        # Make rank(Y) = rY
        for i in range(n):
            b = np.random.randn(n,1)
            Y[:,i,None] = (U @ b) # Each column of Y lies in subspace spanned by cols of U

    else:
        print('Rank Y cannot be less than d.')
    
    Y_vec = Y.flatten('F')

    best_act_cols = brute_cvx_lasso(X,Y, rY)
    if best_act_cols >= rX*rY:
        print('Found one! X: {}, Y: {}'.format(X,Y))
        good_dict = {'X':X, 'Y':Y}
        good_XY.append(good_dict)
    act_cols_arr.append(best_act_cols)
now = datetime.datetime.now().strftime('%m%d%H%M%S')
dest_dir = 'results/brute_cvx_multi_task_lasso/D_{}_N_{}_K_{}_iters_{}_{}'.format(d, n, k, iters, rY, rX, now)
os.makedirs(dest_dir, exist_ok=True)

np.save(os.path.join(dest_dir,'act_cols_arr.npy'), act_cols_arr)
np.save(os.path.join(dest_dir,'good_XY.npy'), good_XY)