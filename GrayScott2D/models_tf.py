# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:22:32 2020

@author: Wsf12
"""

import tensorflow as tf
import numpy as np
import time


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


class ResidualSampler:
    # Initialize the class
    def __init__(self, X, name=None):
        self.X = X
        self.N = self.X.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(self.N, batch_size, replace=False)
        X_batch = self.X[idx, :]
        return X_batch


class DataSampler:
    # Initialize the class
    def __init__(self, X, Y, name=None):
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(self.N, batch_size, replace=False)
        X_batch = self.X[idx, :]
        Y_batch = self.Y[idx, :]
        return X_batch, Y_batch


class Gray_Scott2D:
    # Initialize the class
    def __init__(self, data_sampler, residual_sampler, layers, b, d):

        N = data_sampler.N
        X, U = data_sampler.sample(N)

        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]
        self.mu_y, self.sigma_y = self.mu_X[2], self.sigma_X[2]

        self.mu_U, self.sigma_U = U.mean(0), U.std(0)
        self.mu_u, self.sigma_u = self.mu_U[0], self.sigma_U[0]
        self.mu_v, self.sigma_v = self.mu_U[1], self.sigma_U[1]

        # Samplers
        self.data_sampler = data_sampler
        self.residual_sampler = residual_sampler

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Parameters
        self.epsilon1 = tf.Variable(-10.0, dtype=tf.float32)
        self.epsilon2 = tf.Variable(-10.0, dtype=tf.float32)

        self.b = b
        self.d = d

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.v_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.w_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.u_pred, self.v_pred = self.net_u(self.t_u_tf,
                                              self.x_u_tf,
                                              self.y_u_tf)

        self.u_res_pred, self.v_res_pred = self.net_r(self.t_r_tf,
                                                      self.x_r_tf,
                                                      self.y_r_tf)

        # Data loss
        self.loss_u_data = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_v_data = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
        self.loss_data = self.loss_u_data + self.loss_v_data

        # Residual loss
        self.loss_res_u = tf.reduce_mean(tf.square(self.u_res_pred))
        self.loss_res_v = tf.reduce_mean(tf.square(self.v_res_pred))

        self.loss_res = self.loss_res_u + self.loss_res_v

        # Total loss
        self.loss = self.loss_data + self.loss_res

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                        self.global_step,
                                                        1000, 0.9,
                                                        staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                            global_step=self.global_step)

        # Logger
        self.loss_u_log = []
        self.loss_r_log = []

        self.ep1_log = []
        self.ep2_log = []

        self.saver = tf.train.Saver()

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, H):
        num_layers = len(self.layers)
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    def net_u(self, t, x, y):
        # Compute scalar potentials
        out = self.neural_net(tf.concat([t, x, y], 1))
        u = out[:, 0:1]
        v = out[:, 1:2]

        # De-normalize
        u = u * self.sigma_u + self.mu_u
        v = v * self.sigma_v + self.mu_v

        return u, v

    def net_r(self, t, x, y):
        u, v = self.net_u(t, x, y)

        u_t = tf.gradients(u, t)[0] / self.sigma_t
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y

        v_t = tf.gradients(v, t)[0] / self.sigma_t
        v_x = tf.gradients(v, x)[0] / self.sigma_x
        v_y = tf.gradients(v, y)[0] / self.sigma_y

        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        u_yy = tf.gradients(u_y, y)[0] / self.sigma_y

        v_xx = tf.gradients(v_x, x)[0] / self.sigma_x
        v_yy = tf.gradients(v_y, y)[0] / self.sigma_y

        u_res = u_t - tf.exp(self.epsilon1) * (u_xx + u_yy) - self.b * (1 - u) + u * tf.square(v)
        v_res = v_t - tf.exp(self.epsilon2) * (v_xx + v_yy) + self.d * v - u * tf.square(v)

        return u_res, v_res

    def fetch_minibatch_data(self, N):
        X, Y = self.data_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def fetch_minibatch_residual(self, N):
        X = self.residual_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X

    def train(self, nIter=10000, batch_size=128):
        start_time = time.time()
        for it in range(nIter):
            X_u_batch, U_batch = self.fetch_minibatch_data(batch_size)
            X_r_batch, _ = self.fetch_minibatch_residual(batch_size)

            tf_dict = {self.t_u_tf: X_u_batch[:, 0:1], self.x_u_tf: X_u_batch[:, 1:2], self.y_u_tf: X_u_batch[:, 2:3],
                       self.t_r_tf: X_r_batch[:, 0:1], self.x_r_tf: X_r_batch[:, 1:2], self.y_r_tf: X_r_batch[:, 2:3],
                       self.u_tf: U_batch[:, 0:1], self.v_tf: U_batch[:, 1:2]}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_u_value = self.sess.run(self.loss_data, tf_dict)
                loss_r_value = self.sess.run(self.loss_res, tf_dict)

                ep1_value = self.sess.run(self.epsilon1)
                ep2_value = self.sess.run(self.epsilon2)

                self.loss_u_log.append(loss_u_value)
                self.loss_r_log.append(loss_r_value)

                self.ep1_log.append(np.exp(ep1_value))
                self.ep2_log.append(np.exp(ep2_value))

                print('It: %d, Data: %.3e, Residual: %.3e, Time: %.2f' %
                      (it, loss_u_value, loss_r_value, elapsed))

                print('ep1: {:.3e}, ep2: {:.3e}'.format(np.exp(ep1_value), np.exp(ep2_value)))

                start_time = time.time()

    def predict(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1],
                   self.x_u_tf: X_star[:, 1:2],
                   self.y_u_tf: X_star[:, 2:3]}
        u_pred = self.sess.run(self.u_pred, tf_dict)
        v_pred = self.sess.run(self.v_pred, tf_dict)
        return u_pred, v_pred


class Gray_Scott2D_FF:
    # Initialize the class
    def __init__(self, data_sampler, residual_sampler, layers, b, d):

        N = data_sampler.N
        X, U = data_sampler.sample(N)

        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]
        self.mu_y, self.sigma_y = self.mu_X[2], self.sigma_X[2]

        self.mu_U, self.sigma_U = U.mean(0), U.std(0)
        self.mu_u, self.sigma_u = self.mu_U[0], self.sigma_U[0]
        self.mu_v, self.sigma_v = self.mu_U[1], self.sigma_U[1]

        # Samplers
        self.data_sampler = data_sampler
        self.residual_sampler = residual_sampler

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        #        self.W_t = tf.Variable(tf.random_uniform([1, layers[0] // 2], minval=0, maxval=1), dtype=tf.float32)
        #        self.b_t = tf.Variable(tf.random_uniform([1, layers[0]], dtype=tf.float32), dtype=tf.float32)
        #
        #        self.W_x = tf.Variable(tf.random_uniform([2, layers[0] // 2], minval=0, maxval=20), dtype=tf.float32)
        #        self.b_x = tf.Variable(tf.random_uniform([2, layers[0]], dtype=tf.float32), dtype=tf.float32)

        self.W_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 1, dtype=tf.float32,
                               trainable=False)

        self.W_x = tf.Variable(tf.random_normal([2, layers[0] // 2], dtype=tf.float32) * 30, dtype=tf.float32,
                               trainable=False)

        # Parameters
        self.epsilon1 = tf.Variable(-10.0, dtype=tf.float32)
        self.epsilon2 = tf.Variable(-10.0, dtype=tf.float32)

        self.b = b
        self.d = d

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.v_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.w_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.u_pred, self.v_pred = self.net_u(self.t_u_tf,
                                              self.x_u_tf,
                                              self.y_u_tf)

        self.u_res_pred, self.v_res_pred = self.net_r(self.t_r_tf,
                                                      self.x_r_tf,
                                                      self.y_r_tf)

        # Data loss
        self.loss_u_data = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_v_data = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
        self.loss_data = self.loss_u_data + self.loss_v_data

        # Residual loss
        self.loss_res_u = tf.reduce_mean(tf.square(self.u_res_pred))
        self.loss_res_v = tf.reduce_mean(tf.square(self.v_res_pred))

        self.loss_res = self.loss_res_u + self.loss_res_v

        # Total loss
        self.loss = self.loss_data + self.loss_res

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                        self.global_step,
                                                        1000, 0.9,
                                                        staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                            global_step=self.global_step)

        # Logger
        self.loss_u_log = []
        self.loss_r_log = []

        self.ep1_log = []
        self.ep2_log = []

        self.saver = tf.train.Saver()

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []

        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.random_normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)

        W = self.xavier_init(size=[layers[-2], layers[-1]])
        b = tf.Variable(tf.random_normal([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)

        return weights, biases

    def neural_net(self, H):
        num_layers = len(self.layers)
        t = H[:, 0:1]
        x = H[:, 1:3]

        H_t = tf.concat([tf.sin(tf.matmul(t, self.W_t)),
                         tf.cos(tf.matmul(t, self.W_t))], 1)  # (N ,100))

        H_x = tf.concat([tf.sin(tf.matmul(x, self.W_x)),
                         tf.cos(tf.matmul(x, self.W_x))], 1)

        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]

            H_t = tf.tanh(tf.add(tf.matmul(H_t, W), b))
            H_x = tf.tanh(tf.add(tf.matmul(H_x, W), b))

        H = tf.multiply(H_t, H_x)

        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    def net_u(self, t, x, y):
        # Compute scalar potentials
        out = self.neural_net(tf.concat([t, x, y], 1))
        u = out[:, 0:1]
        v = out[:, 1:2]

        # De-normalize
        u = u * self.sigma_u + self.mu_u
        v = v * self.sigma_v + self.mu_v

        return u, v

    def net_r(self, t, x, y):
        u, v = self.net_u(t, x, y)

        u_t = tf.gradients(u, t)[0] / self.sigma_t
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y

        v_t = tf.gradients(v, t)[0] / self.sigma_t
        v_x = tf.gradients(v, x)[0] / self.sigma_x
        v_y = tf.gradients(v, y)[0] / self.sigma_y

        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        u_yy = tf.gradients(u_y, y)[0] / self.sigma_y

        v_xx = tf.gradients(v_x, x)[0] / self.sigma_x
        v_yy = tf.gradients(v_y, y)[0] / self.sigma_y

        u_res = u_t - tf.exp(self.epsilon1) * (u_xx + u_yy) - self.b * (1 - u) + u * tf.square(v)
        v_res = v_t - tf.exp(self.epsilon2) * (v_xx + v_yy) + self.d * v - u * tf.square(v)

        #        u_res = u_t - self.epsilon1 * (u_xx + u_yy) - self.b * (1 - u) + u * tf.square(v)
        #        v_res = v_t - self.epsilon2 * (v_xx + v_yy) + self.d * v - u * tf.square(v)

        return u_res, v_res

    def fetch_minibatch_data(self, N):
        X, Y = self.data_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def fetch_minibatch_residual(self, N):
        X, Y = self.residual_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = time.time()
        for it in range(nIter):
            X_u_batch, U_batch = self.fetch_minibatch_data(batch_size)
            X_r_batch, _ = self.fetch_minibatch_residual(batch_size)

            tf_dict = {self.t_u_tf: X_u_batch[:, 0:1], self.x_u_tf: X_u_batch[:, 1:2], self.y_u_tf: X_u_batch[:, 2:3],
                       self.t_r_tf: X_r_batch[:, 0:1], self.x_r_tf: X_r_batch[:, 1:2], self.y_r_tf: X_r_batch[:, 2:3],
                       self.u_tf: U_batch[:, 0:1], self.v_tf: U_batch[:, 1:2]}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_u_value = self.sess.run(self.loss_data, tf_dict)
                loss_r_value = self.sess.run(self.loss_res, tf_dict)

                ep1_value = self.sess.run(self.epsilon1)
                ep2_value = self.sess.run(self.epsilon2)

                self.loss_u_log.append(loss_u_value)
                self.loss_r_log.append(loss_r_value)

                self.ep1_log.append(np.exp(ep1_value))
                self.ep2_log.append(np.exp(ep2_value))

                print('It: %d, Data: %.3e, Residual: %.3e, Time: %.2f' %
                      (it, loss_u_value, loss_r_value, elapsed))

                print('ep1: {:.3e}, ep2: {:.3e}'.format(np.exp(ep1_value), np.exp(ep2_value)))
                #                print('ep1: {:.3e}, ep2: {:.3e}'.format(ep1_value, ep2_value))

                start_time = time.time()

    def predict(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1],
                   self.x_u_tf: X_star[:, 1:2],
                   self.y_u_tf: X_star[:, 2:3]}
        u_pred = self.sess.run(self.u_pred, tf_dict)
        v_pred = self.sess.run(self.v_pred, tf_dict)
        return u_pred, v_pred


class Gray_Scott2D_ST_mFF:
    # Initialize the class
    def __init__(self, data_sampler, residual_sampler, layers, b, d):

        N = data_sampler.N
        X, U = data_sampler.sample(N)

        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]
        self.mu_y, self.sigma_y = self.mu_X[2], self.sigma_X[2]

        self.mu_U, self.sigma_U = U.mean(0), U.std(0)
        self.mu_u, self.sigma_u = self.mu_U[0], self.sigma_U[0]
        self.mu_v, self.sigma_v = self.mu_U[1], self.sigma_U[1]

        # Samplers
        self.data_sampler = data_sampler
        self.residual_sampler = residual_sampler

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        self.W_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 1, dtype=tf.float32,
                               trainable=False)

        self.W1_x = tf.Variable(tf.random_normal([2, layers[0] // 2], dtype=tf.float32) * 1, dtype=tf.float32,
                                trainable=False)
        self.W2_x = tf.Variable(tf.random_normal([2, layers[0] // 2], dtype=tf.float32) * 10, dtype=tf.float32,
                                trainable=False)
        self.W3_x = tf.Variable(tf.random_normal([2, layers[0] // 2], dtype=tf.float32) * 50, dtype=tf.float32,
                                trainable=False)

        # Parameters
        #        self.epsilon1 = epsilon1
        #        self.epsilon2 = epsilon2

        self.epsilon1 = tf.Variable(-10.0, dtype=tf.float32)
        self.epsilon2 = tf.Variable(-10.0, dtype=tf.float32)
        self.b = b
        self.d = d

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.v_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.w_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.u_pred, self.v_pred = self.net_u(self.t_u_tf,
                                              self.x_u_tf,
                                              self.y_u_tf)

        self.u_res_pred, self.v_res_pred = self.net_r(self.t_r_tf,
                                                      self.x_r_tf,
                                                      self.y_r_tf)

        # Data loss
        self.loss_u_data = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_v_data = tf.reduce_mean(tf.square(self.v_tf - self.v_pred))
        self.loss_data = self.loss_u_data + self.loss_v_data

        # Residual loss
        self.loss_res_u = tf.reduce_mean(tf.square(self.u_res_pred))
        self.loss_res_v = tf.reduce_mean(tf.square(self.v_res_pred))

        self.loss_res = self.loss_res_u + self.loss_res_v

        # Total loss
        self.loss = self.loss_data + self.loss_res

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                        self.global_step,
                                                        5000, 0.9,
                                                        staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                            global_step=self.global_step)

        # Logger
        self.loss_u_log = []
        self.loss_r_log = []

        self.ep1_log = []
        self.ep2_log = []

        self.saver = tf.train.Saver()

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []

        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.random_normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)

        W = self.xavier_init(size=[3 * layers[-2], layers[-1]])
        b = tf.Variable(tf.random_normal([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)

        return weights, biases

    def neural_net(self, H):
        num_layers = len(self.layers)
        t = H[:, 0:1]
        x = H[:, 1:3]

        H_t = tf.concat([tf.sin(tf.matmul(t, self.W_t)),
                         tf.cos(tf.matmul(t, self.W_t))], 1)  # (N ,100))

        H1_x = tf.concat([tf.sin(tf.matmul(x, self.W1_x)),
                          tf.cos(tf.matmul(x, self.W1_x))], 1)

        H2_x = tf.concat([tf.sin(tf.matmul(x, self.W2_x)),
                          tf.cos(tf.matmul(x, self.W2_x))], 1)

        H3_x = tf.concat([tf.sin(tf.matmul(x, self.W3_x)),
                          tf.cos(tf.matmul(x, self.W3_x))], 1)

        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]

            H_t = tf.tanh(tf.add(tf.matmul(H_t, W), b))

            H1_x = tf.tanh(tf.add(tf.matmul(H1_x, W), b))
            H2_x = tf.tanh(tf.add(tf.matmul(H2_x, W), b))
            H3_x = tf.tanh(tf.add(tf.matmul(H3_x, W), b))

        H1 = tf.multiply(H_t, H1_x)
        H2 = tf.multiply(H_t, H2_x)
        H3 = tf.multiply(H_t, H3_x)

        H = tf.concat([H1, H2, H3], 1)

        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    def net_u(self, t, x, y):
        # Compute scalar potentials
        out = self.neural_net(tf.concat([t, x, y], 1))
        u = out[:, 0:1]
        v = out[:, 1:2]

        # De-normalize
        u = u * self.sigma_u + self.mu_u
        v = v * self.sigma_v + self.mu_v

        return u, v

    def net_r(self, t, x, y):
        u, v = self.net_u(t, x, y)

        u_t = tf.gradients(u, t)[0] / self.sigma_t
        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_y = tf.gradients(u, y)[0] / self.sigma_y

        v_t = tf.gradients(v, t)[0] / self.sigma_t
        v_x = tf.gradients(v, x)[0] / self.sigma_x
        v_y = tf.gradients(v, y)[0] / self.sigma_y

        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x
        u_yy = tf.gradients(u_y, y)[0] / self.sigma_y

        v_xx = tf.gradients(v_x, x)[0] / self.sigma_x
        v_yy = tf.gradients(v_y, y)[0] / self.sigma_y

        u_res = u_t - tf.exp(self.epsilon1) * (u_xx + u_yy) - self.b * (1 - u) + u * tf.square(v)
        v_res = v_t - tf.exp(self.epsilon2) * (v_xx + v_yy) + self.d * v - u * tf.square(v)

        return u_res, v_res

    def fetch_minibatch_data(self, N):
        X, Y = self.data_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def fetch_minibatch_residual(self, N):
        X, Y = self.residual_sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    def train(self, nIter=10000, batch_size=128):
        start_time = time.time()
        for it in range(nIter):
            X_u_batch, U_batch = self.fetch_minibatch_data(batch_size)
            X_r_batch, _ = self.fetch_minibatch_residual(batch_size)

            tf_dict = {self.t_u_tf: X_u_batch[:, 0:1], self.x_u_tf: X_u_batch[:, 1:2], self.y_u_tf: X_u_batch[:, 2:3],
                       self.t_r_tf: X_r_batch[:, 0:1], self.x_r_tf: X_r_batch[:, 1:2], self.y_r_tf: X_r_batch[:, 2:3],
                       self.u_tf: U_batch[:, 0:1], self.v_tf: U_batch[:, 1:2]}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_u_value = self.sess.run(self.loss_data, tf_dict)
                loss_r_value = self.sess.run(self.loss_res, tf_dict)

                ep1_value = self.sess.run(self.epsilon1)
                ep2_value = self.sess.run(self.epsilon2)

                self.loss_u_log.append(loss_u_value)
                self.loss_r_log.append(loss_r_value)

                self.ep1_log.append(np.exp(ep1_value))
                self.ep2_log.append(np.exp(ep2_value))

                print('It: %d, Data: %.3e, Residual: %.3e, Time: %.2f' %
                      (it, loss_u_value, loss_r_value, elapsed))

                print('ep1: {:.3e}, ep2: {:.3e}'.format(np.exp(ep1_value), np.exp(ep2_value)))

                start_time = time.time()

    def predict(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1],
                   self.x_u_tf: X_star[:, 1:2],
                   self.y_u_tf: X_star[:, 2:3]}
        u_pred = self.sess.run(self.u_pred, tf_dict)
        v_pred = self.sess.run(self.v_pred, tf_dict)
        return u_pred, v_pred


