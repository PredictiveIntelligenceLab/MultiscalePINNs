# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:17:33 2020

@author: Wsf12
"""

import tensorflow as tf
from Compute_Jacobian import jacobian
import numpy as np
import timeit

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y

class NN:
    def __init__(self, layers, bcs_samplers, res_samplers, u, a, b, sigma):

        # Normalize the input
        X, _ = res_samplers.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        
        # Samplers
        self.bcs_samplers = bcs_samplers
        self.res_samplers = res_samplers

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.u_bc1_pred = self.net_u(self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.x_bc2_tf)

        self.u_pred = self.net_u(self.x_u_tf)
        self.r_pred = self.net_r(self.x_r_tf)

        # Boundary loss
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred - self.u_bc1_tf))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred - self.u_bc2_tf))

        self.loss_bcs = self.loss_bc1 + self.loss_bc2

        # Residual loss        
        self.loss_res = tf.reduce_mean(tf.square(self.r_tf - self.r_pred))
        
        # Total loss
        self.loss = self.loss_res + self.loss_bcs
        
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

        # Test data
        N_test = 1000
        
        self.X_star = np.linspace(0, 1, N_test)[:, None]
        self.u_star = u(self.X_star, a,b)

        # Logger
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.l2_error_log = []

        # Saver
        self.saver = tf.train.Saver()

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
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
        
    def forward_pass(self, H):
        num_layers = len(self.layers)
        
        for l in range(0, num_layers - 2): # number_layers  - 1?
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
        
    def net_u(self, x):
        u = self.forward_pass(x)
        return u

    # Forward pass for f
    def net_r(self, x):
        u = self.net_u(x)

        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x

        res_u = u_xx
        return res_u

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128, log_NTK=True, log_weights=True):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_samplers[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_samplers[1], batch_size)

            # Fetch residual mini-batch
            X_res_batch, f_batch = self.fetch_minibatch(self.res_samplers,  batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_bc1_tf: X_bc1_batch, self.x_bc2_tf: X_bc2_batch,
                       self.u_bc1_tf: u_bc1_batch, self.u_bc2_tf: u_bc2_batch,
                       self.x_u_tf: X_res_batch, self.x_r_tf: X_res_batch,
                       self.r_tf: f_batch
                       }
        
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value, loss_res_value = self.sess.run([self.loss_bcs, self.loss_res], tf_dict)
                
                u_pred = self.predict_u(self.X_star)
                error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.l2_error_log.append(error_u)

                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_res: %.3e ,Time: %.2f' %
                      (it, loss_value, loss_bcs_value, loss_res_value, elapsed))

                start_time = timeit.default_timer()

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_r_tf: X_star}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star


class NN_FF:
    def __init__(self, layers, bcs_samplers, res_samplers, u, a, b, sigma):

        # Normalize the input
        X, _ = res_samplers.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        
        # Samplers
        self.bcs_samplers = bcs_samplers
        self.res_samplers = res_samplers

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize Fourier features
        self.W = tf.Variable(tf.random_normal([1, layers[0] //2], dtype=tf.float32) * sigma, dtype=tf.float32, trainable=False)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.u_bc1_pred = self.net_u(self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.x_bc2_tf)

        self.u_pred = self.net_u(self.x_u_tf)
        self.r_pred = self.net_r(self.x_r_tf)

        # Boundary loss
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred - self.u_bc1_tf))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred - self.u_bc2_tf))

        self.loss_bcs = self.loss_bc1 + self.loss_bc2

        # Residual loss        
        self.loss_res = tf.reduce_mean(tf.square(self.r_tf - self.r_pred))
        
        # Total loss
        self.loss = self.loss_res + self.loss_bcs
        
        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        100, 0.99, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # Test data
        N_test = 1000
        self.X_star = np.linspace(0, 1, N_test)[:, None]
        self.u_star = u(self.X_star, a, b)
        self.l2_error_log = []
        
        # Logger
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.l2_error_log = []

        # Saver
        self.saver = tf.train.Saver()

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
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
        
    def forward_pass(self, H):
        num_layers = len(self.layers)

        # Fourier feature encoding
        H = tf.concat([tf.sin(tf.matmul(H, self.W)),
                       tf.cos(tf.matmul(H, self.W))], 1) 

        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
        
    def net_u(self, x):
        u = self.forward_pass(x)
        return u

    # Forward pass for f
    def net_r(self, x):
        u = self.net_u(x)

        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x

        res_u = u_xx
        return res_u
    
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
        D = x1.shape[0]
        N = len(J1_list)
        
        Ker = tf.zeros((D,D))
        for k in range(N):
            J1 = tf.reshape(J1_list[k], shape=(D,-1))
            J2 = tf.reshape(J2_list[k], shape=(D,-1))
            
            K = tf.matmul(J1, tf.transpose(J2))
            Ker = Ker + K
        return Ker

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128, log_NTK=True, log_weights=True):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_samplers[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_samplers[1], batch_size)

            # Fetch residual mini-batch
            X_res_batch, f_batch = self.fetch_minibatch(self.res_samplers,  batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_bc1_tf: X_bc1_batch, self.x_bc2_tf: X_bc2_batch,
                       self.u_bc1_tf: u_bc1_batch, self.u_bc2_tf: u_bc2_batch,
                       self.x_u_tf: X_res_batch, self.x_r_tf: X_res_batch,
                       self.r_tf: f_batch
                       }
        
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value, loss_res_value = self.sess.run([self.loss_bcs, self.loss_res], tf_dict)
                
                u_pred = self.predict_u(self.X_star)
                error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.l2_error_log.append(error_u)

                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_res: %.3e ,Time: %.2f' %
                      (it, loss_value, loss_bcs_value, loss_res_value, elapsed))

                start_time = timeit.default_timer()

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_r_tf: X_star}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star

   
class NN_mFF:
    def __init__(self, layers, bcs_samplers, res_samplers, u,a, b, sigma):

        # Normalize the input
        X, _ = res_samplers.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        
        # Samplers
        self.bcs_samplers = bcs_samplers
        self.res_samplers = res_samplers

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize Fourier features
        self.W1 = tf.Variable(tf.random_normal([1, layers[0] //2], dtype=tf.float32) * 1, dtype=tf.float32, trainable=False)
        self.W2 = tf.Variable(tf.random_normal([1, layers[0] //2], dtype=tf.float32) * sigma, dtype=tf.float32, trainable=False)

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        # Evaluate predictions
        self.u_bc1_pred = self.net_u(self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.x_bc2_tf)

        self.u_pred = self.net_u(self.x_u_tf)
        self.r_pred = self.net_r(self.x_r_tf)

        # Boundary loss
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred - self.u_bc1_tf))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred - self.u_bc2_tf))

        self.loss_bcs = self.loss_bc1 + self.loss_bc2

        # Residual loss        
        self.loss_res = tf.reduce_mean(tf.square(self.r_tf - self.r_pred))
        
        # Total loss
        self.loss = self.loss_res + self.loss_bcs
        
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
        
        # Test data
        N_test = 1000
        self.X_star = np.linspace(0, 1, N_test)[:, None]
        self.u_star = u(self.X_star, a,b)
        
        self.l2_error_log = []
        
        # Logger
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.saver = tf.train.Saver()

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        
        num_layers = len(layers)
    
        for l in range(0, num_layers - 2):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.random_normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        
        W = self.xavier_init(size=[2 * layers[-2], layers[-1]])
        b = tf.Variable(tf.random_normal([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
        
        return weights, biases
    
    def forward_pass(self, H):
        num_layers = len(self.layers)

        # Fourier feature encodings
        H1 = tf.concat([tf.sin(tf.matmul(H, self.W1)),
                        tf.cos(tf.matmul(H, self.W1))], 1)
        H2 = tf.concat([tf.sin(tf.matmul(H, self.W2)),
                        tf.cos(tf.matmul(H, self.W2))], 1)   # H1  (N ,50))

        for l in range(0, num_layers-2):
            W = self.weights[l]
            b = self.biases[l]
            H1 = tf.tanh(tf.add(tf.matmul(H1, W), b))
            
            W = self.weights[l]
            b = self.biases[l]
            H2 = tf.tanh(tf.add(tf.matmul(H2, W), b))

        # Concatenate the network outputs
        H = tf.concat([H1, H2], 1)
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
        
    def net_u(self, x):
        u = self.forward_pass(x)
        return u

    # Forward pass for f
    def net_r(self, x):
        u = self.net_u(x)

        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x

        res_u = u_xx
        return res_u

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128, log_NTK=True, log_weights=True):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_samplers[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_samplers[1], batch_size)

            # Fetch residual mini-batch
            X_res_batch, f_batch = self.fetch_minibatch(self.res_samplers,  batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_bc1_tf: X_bc1_batch, self.x_bc2_tf: X_bc2_batch,
                       self.u_bc1_tf: u_bc1_batch, self.u_bc2_tf: u_bc2_batch,
                       self.x_u_tf: X_res_batch, self.x_r_tf: X_res_batch,
                       self.r_tf: f_batch
                       }
        
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value, loss_res_value = self.sess.run([self.loss_bcs, self.loss_res], tf_dict)

                u_pred = self.predict_u(self.X_star)
                error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.l2_error_log.append(error_u)

                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_res: %.3e ,Time: %.2f' %
                      (it, loss_value, loss_bcs_value, loss_res_value, elapsed))

                start_time = timeit.default_timer()

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_r_tf: X_star}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star


