# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:20:01 2020

@author: sifan
"""

import tensorflow as tf
from Compute_Jacobian import jacobian
import numpy as np
import timeit

# Data Sampler
class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    # Sample function
    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y
    

class NN_FF:
    def __init__(self, layers, X_u, Y_u, a, u, sigma):

        """
        :param layers: Layers of the network
        :param X_u, Y_u: Training data
        :param a:  Hyper-parameter of the target function
        :param u:  the target function
        :param sigma: Hyper-parameter of the Fourier features
        """

        self.mu_X, self.sigma_X = X_u.mean(0), X_u.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]

        # Normalize the input of the network
        self.X_u = (X_u - self.mu_X) / self.sigma_X
        self.Y_u = Y_u

        # Initialize Fourier features
        self.W = tf.Variable(tf.random_normal([1, layers[0] //2], dtype=tf.float32) * sigma, dtype=tf.float32, trainable=False)

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
            
        # Define the size of the Kernel
        self.D_u = X_u.shape[0]
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_ntk_tf = tf.placeholder(tf.float32, shape=(self.D_u, 1))

        # Evaluate predictions
        self.u_pred = self.net_u(self.x_u_tf)

        # Evaluate NTK predictions
        self.u_ntk_pred = self.net_u(self.x_u_ntk_tf)
     
        # Boundary loss
        self.loss_u = tf.reduce_mean(tf.square(self.u_pred - self.u_tf))   
        
        # Total loss
        self.loss = self.loss_u

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)

        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Model Saver
        self.saver = tf.train.Saver()

        # Compute the Jacobian for weights and biases in each hidden layer
        self.J_u = self.compute_jacobian(self.u_ntk_pred)

        # The empirical NTK = J J^T, compute NTK of PINNs
        self.K = self.compute_ntk(self.J_u, self.x_u_ntk_tf, self.J_u, self.x_u_ntk_tf)

        # Loss Logger
        self.loss_u_log = []

        # NTK logger
        self.K_log = []

        # Weights logger
        self.weights_log = []
        self.biases_log = []

        # Training error and test error
        N_train  = 100
        N_test = 1000

        # Training data
        self.X_train = np.linspace(0, 1, N_train)[:, None]
        self.Y_train = u(self.X_train, a)

        # Test data
        self.X_test = np.linspace(0, 1, N_test)[:, None]
        self.Y_test = u(self.X_test, a)

        # Error loggers
        self.train_error_log = []
        self.test_error_log = []

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # NTK initialization
    def NTK_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        std = 1. / np.sqrt(in_dim)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * std,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.random_normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluate the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        
        H = tf.concat([tf.sin(tf.matmul(H, self.W)),
                       tf.cos(tf.matmul(H, self.W))], 1) 

        for l in range(0, num_layers - 2): # number_layers  - 1?
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Define the neural net
    def net_u(self, x):
        u = self.forward_pass(x)
        return u

    # Compute Jacobian for each weights and biases in each layer and retrun a list
    def compute_jacobian(self, f):
        J_list =[]
        L = len(self.weights)
        for i in range(L):
            J_w = jacobian(f, self.weights[i])
            J_list.append(J_w)

        for i in range(L):
            J_b = jacobian(f, self.biases[i])
            J_list.append(J_b)
        return J_list

    # Compute the empirical NTK = J J^T
    def compute_ntk(self, J1_list, x1, J2_list, x2):
        D1 = x1.shape[0]
        D2 = x2.shape[0]
        N = len(J1_list)

        Ker = tf.zeros((D1, D2))
        for k in range(N):
            J1 = tf.reshape(J1_list[k], shape=(D1, -1))
            J2 = tf.reshape(J2_list[k], shape=(D2, -1))

            K = tf.matmul(J1, tf.transpose(J2))
            Ker = Ker + K
        return Ker

    # Fetch minibatch
    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, log_NTK=True, log_weights=True):

        start_time = timeit.default_timer()

        for it in range(nIter):
            # Fetch  mini-batches
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_u_tf: self.X_u, self.u_tf: self.Y_u
                       }

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time

                loss_value = self.sess.run(self.loss, tf_dict)
                loss_u_value = self.sess.run(self.loss_u, tf_dict)

                # Store the loss values
                self.loss_u_log.append(loss_u_value)

                # Compute the training error
                u_pred_train = self.predict_u(self.X_train)
                training_error = np.linalg.norm(self.Y_train - u_pred_train, 2) / np.linalg.norm(self.Y_train, 2)

                # Compute the test error
                u_pred_test = self.predict_u(self.X_test)
                test_error = np.linalg.norm(self.Y_test - u_pred_test, 2) / np.linalg.norm(self.Y_test, 2)

                # Store the training and test errors
                self.train_error_log.append(training_error)
                self.test_error_log.append(test_error)

                # print the loss values
                print('It: %d, Loss: %.3e, Loss_bcs: %.3e,Time: %.2f' %
                      (it, loss_value, loss_u_value, elapsed))

                start_time = timeit.default_timer()

            # Store the NTK matrix for every 100 iterations
            if log_NTK:
                # provide x, x' for NTK
                if it % 100 == 0:
                    print("Compute NTK...")
                    tf_dict = {self.x_u_ntk_tf: self.X_u}
                    K_value = self.sess.run(self.K, tf_dict)
                    self.K_log.append(K_value)

            # Store the weights and biases of the network for every 100 iterations
            if log_weights:
                if it % 100 ==0:
                    print("Weights stored...")
                    weights = self.sess.run(self.weights)
                    biases = self.sess.run(self.biases)
                    
                    self.weights_log.append(weights)
                    self.biases_log.append(biases)
                
    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star


