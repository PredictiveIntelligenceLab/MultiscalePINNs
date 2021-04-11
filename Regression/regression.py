# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:20:08 2020

@author: sifan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models_tf import Sampler, NN_FF


if __name__ == '__main__':

    # Define solution and its Laplace
    def u(x, a):
        return np.sin(np.pi * x) +  np.cos(np.pi * a * x)

    # Define computational domain
    dom_coords = np.array([[0.0],
                           [1.0]])

    # Training data on u(x) 
    N_u = 100
    X_u = np.linspace(dom_coords[0, 0],
                      dom_coords[1, 0], N_u)[:, None]

    a = 10 
    Y_u = u(X_u, a)
    
    # Test data
    nn = 1000
    X_star = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    u_star = u(X_star, a)
    
    # Define the model
    layers = [100, 100, 100, 1]
    sigma = 10   # Hyper-parameter of the Fourier features
    model = NN_FF(layers, X_u, Y_u, a, u,  sigma)

    # Train the model for different epochs
    epoch_list = [10, 90, 900]  # 1000 iterations in total
    u_pred_list = []

    for epoch in epoch_list:
       # Train the model
       model.train(nIter=epoch, log_NTK=True, log_weights=True)
       
       # Predictions
       u_pred = model.predict_u(X_star)
       u_pred_list.append(u_pred)

    # Evaulate the relative l2 error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Relative L2 error_u: {:.2e}'.format(error_u))

    # Create loggers for the eigenvalues of the NTK
    lambda_K_log = []

    # Restore the NTK
    K_list = model.K_log

    for k in range(len(K_list)):
        K = K_list[k]

        # Compute eigenvalues
        lambda_K, eigvec_K = np.linalg.eig(K)
        
        # Sort in descresing order
        lambda_K = np.sort(np.real(lambda_K))[::-1]
        
        # Store eigenvalues
        lambda_K_log.append(lambda_K)
        
    # Change of the NTK
    kernel_diff_list = []
    K0 = K_list[0]
    for K in K_list:
        diff = np.linalg.norm(K - K0) / np.linalg.norm(K0) 
        kernel_diff_list.append(diff)

    #######################
    #######################
    
    # Change of the weights
    def compute_weights_diff(weights_1, weights_2):
        weights = []
        N = len(weights_1)
        for k in range(N):
            weight = weights_1[k] - weights_2[k]
            weights.append(weight)
        return weights
    
    def compute_weights_norm(weights, biases):
        norm = 0
        for w in weights:
            norm = norm + np.sum(np.square(w))
        for b in biases:
            norm = norm + np.sum(np.square(b))
        norm = np.sqrt(norm)
        return norm
    
    # Restore the list weights and biases
    weights_log = model.weights_log
    biases_log = model.biases_log

    # The weights and biases at initialization
    weights_0 = weights_log[0]
    biases_0 = biases_log[0]
    
    weights_init_norm = compute_weights_norm(weights_0, biases_0)

    weights_change_list = []

    # Compute the change of weights and biases of the network
    N = len(weights_log)
    for k in range(N):
        weights_diff = compute_weights_diff(weights_log[k], weights_log[0])
        biases_diff = compute_weights_diff(biases_log[k], biases_log[0])
        
        weights_diff_norm = compute_weights_norm(weights_diff, biases_diff)
        weights_change = weights_diff_norm / weights_init_norm
        weights_change_list.append(weights_change)
    

    #################################
    ############## PLot##############
    #################################

    
    # Model predictions
    fig = plt.figure(1, figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(X_u, Y_u, 'o', label='Exact')
    plt.plot(X_star, u_pred, '--', label='u_pred')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(X_star, u_star - u_pred, label='Error')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Eigenvalues of NTK
    fig = plt.figure(2, figsize=(6, 5))
    plt.plot(lambda_K_log[0], label = 'n=0')
    plt.plot(lambda_K_log[-1], '--', label = 'n=40,000')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('index')
    plt.ylabel(r'$\lambda_{uu}$')
    plt.title(r'Eigenvalues of ${K}_{uu}$')
    plt.legend()
    plt.show()

    # Loss values
    loss_u = model.loss_u_log
    fig_3 = plt.figure(3, figsize=(6,5))
    plt.plot(loss_u, label='$\mathcal{L}_{u_b}$')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    


    # Visualize the eigenvectors of the NTK
    fig = plt.figure(figsize=(12, 6))
    with sns.axes_style("darkgrid"):
        plt.subplot(2,3,1)
        plt.plot(X_u,  np.real(eigvec_K[:,0]))
        plt.tight_layout()
        
        plt.subplot(2,3,2)
        plt.plot(X_u,  np.real(eigvec_K[:,1]))
        plt.tight_layout()
        
        plt.subplot(2,3,3)
        plt.plot(X_u,  np.real(eigvec_K[:,2]))
        plt.tight_layout()
        
        plt.subplot(2,3,4)
        plt.plot(X_u,  np.real(eigvec_K[:,3]))
        plt.tight_layout()
    
        plt.subplot(2,3,5)
        plt.plot(X_u,  np.real(eigvec_K[:,4]))
        plt.tight_layout()
        
        plt.subplot(2,3,6)
        plt.plot(X_u,  np.real(eigvec_K[:,5]))
    
        plt.tight_layout()
        plt.show()
    
    # Visualize the eigenvalues of the NTK
    fig = plt.figure(figsize=(6, 5))
    with sns.axes_style("darkgrid"):
        plt.plot(lambda_K_log[0], label=r'$\sigma={}$'.format(sigma))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('index')
        plt.ylabel(r'$\lambda$') 
        plt.title('Spectrum')
        plt.tight_layout()
        plt.legend()
        plt.show()
        
        
    # Model predictions at different epoch
    fig = plt.figure(figsize=(12,4))
    with sns.axes_style("darkgrid"):
        plt.subplot(1,3,1)
        plt.plot(X_u, Y_u, 'o')
        plt.plot(X_star,  u_star, color = 'C0', alpha=0.4, linewidth=6)
        plt.plot(X_star,  u_pred_list[0], color='C3', linestyle='--')
        plt.title('Epoch = 10')
        plt.tight_layout()
        
        plt.subplot(1,3,2)
        plt.plot(X_u, Y_u, 'o')
        plt.plot(X_star,  u_star, color = 'C0', alpha=0.4, linewidth=6)
        plt.plot(X_star,  u_pred_list[1], color='C3', linestyle='--')
        plt.title('Epoch = 100')
        plt.tight_layout()
        
        plt.subplot(1,3,3)
        plt.plot(X_u, Y_u, 'o')
        plt.plot(X_star,  u_star, color = 'C0', alpha=0.4, linewidth=6)
        plt.plot(X_star,  u_pred_list[2], color='C3', linestyle='--')
        plt.title('Epoch = 200')
        plt.tight_layout()
        plt.show()

    
    
     