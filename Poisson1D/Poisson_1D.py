# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:52:42 2020

@author: Wsf12
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns
from models_tf import Sampler, NN, NN_FF, NN_mFF


if __name__ == '__main__':

    # Hyper-parameters
    a = 2
    b = 50

    # Exact solution
    def u(x, a, b):
        return np.sin(np.pi * a * x) + 0.1 * np.sin(np.pi * b * x)

    # Exact PDE residual
    def u_xx(x, a, b):
        return - (np.pi * a) ** 2 * np.sin(np.pi * a * x) - 0.1 * (np.pi * b) ** 2 * np.sin(np.pi * b * x)

    # Define computational domain
    bc1_coords = np.array([[0.0],
                           [0.0]])

    bc2_coords = np.array([[1.0],
                           [1.0]])

    dom_coords = np.array([[0.0],
                           [1.0]])

    # Create boundary sampler
    bc1 = Sampler(1, bc1_coords, lambda x: u(x, a, b), name='Dirichlet BC1')
    bc2 = Sampler(1, bc2_coords, lambda x: u(x, a, b), name='Dirichlet BC2')

    bcs_samplers = [bc1, bc2]

    # Create residual sampler
    res_samplers = Sampler(1, dom_coords, lambda x: u_xx(x, a, b), name='Forcing')

    # Define model
    # For NN model, please use layers = [1, 100, 100, 1]
    layers = [100, 100, 1]
    
    # Hyper-parameter for Fourier features
    sigma = 10
    
    # NN: Vanilla MLP
    # NN_FF : Vanilla Fourier feature network
    # NN_mFF : Multi-scale Fourier feature network
    model = NN(layers, bcs_samplers, res_samplers, u, a, b, sigma)

    # Train model
    model.train(nIter=40000, batch_size=128, log_NTK=False, log_weights=False)

    # Create test data
    nn = 10000
    X_star = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    u_star = u(X_star, a, b)
    r_star = u_xx(X_star, a, b)

    # Predictions
    u_pred = model.predict_u(X_star)
    r_pred = model.predict_r(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_r = np.linalg.norm(r_star - r_pred, 2) / np.linalg.norm(r_star, 2)

    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_r: {:.2e}'.format(error_r))
            
    loss_bcs = model.loss_bcs_log
    loss_res = model.loss_res_log
    l2_error = model.l2_error_log
    
    # Plot
    fig = plt.figure(figsize=(18, 5))
    with sns.axes_style("darkgrid"):
        plt.subplot(1, 3, 1)
        plt.plot(X_star, u_star, label='Exact')
        plt.plot(X_star, u_pred, '--', label='Predicted')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.legend(fontsize=20, loc='upper left')
        plt.tight_layout()

        plt.subplot(1, 3, 2)
        plt.plot(X_star, u_star - u_pred, label='Error')
        plt.xlabel('$x$')
        plt.ylabel('Point-wise error')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        iters = 100 * np.arange(len(loss_res))

        plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$', linewidth=2)
        plt.plot(iters, loss_bcs, label='$\mathcal{L}_{b}$', linewidth=2)
        plt.plot(iters, l2_error, label=r'$L^2$ error', linewidth=2)

        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), fontsize=20)
        plt.tight_layout()
        plt.show()















