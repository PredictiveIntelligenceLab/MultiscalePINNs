import tensorflow as tf
from Compute_Jacobian import jacobian
import numpy as np
import timeit

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name = None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
    def sample(self, N):
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y


class Wave1D_NTK:
    # Plain MLP with NTK adaptive weights

    # Initialize the class
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, c, kernel_size, X_star, u_star):

        # Normalize input
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # Test data
        self.X_star = X_star
        self.u_star = u_star

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Initialize weights for losses
        self.lambda_u_val = np.array(1.0)
        self.lambda_ut_val = np.array(1.0)
        self.lambda_r_val = np.array(1.0)
      
        # Wave velocity
        self.c = tf.constant(c, dtype=tf.float32)

        # Size of the NTK
        self.kernel_size = kernel_size

        D1 = self.kernel_size    # size of K_u
        D2 = self.kernel_size    # size of K_ut
        D3 = self.kernel_size    # size of K_r

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.lambda_u_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)
        self.lambda_ut_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)
        self.lambda_r_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)

        self.t_u_ntk_tf = tf.placeholder(tf.float32, shape=(D1, 1))
        self.x_u_ntk_tf = tf.placeholder(tf.float32, shape=(D1, 1))
        
        self.t_ut_ntk_tf = tf.placeholder(tf.float32, shape=(D2, 1))
        self.x_ut_ntk_tf = tf.placeholder(tf.float32, shape=(D2, 1))
        
        self.t_r_ntk_tf = tf.placeholder(tf.float32, shape=(D3, 1))
        self.x_r_ntk_tf = tf.placeholder(tf.float32, shape=(D3, 1))

        # Evaluate predictions
        self.u_ics_pred = self.net_u(self.t_ics_tf, self.x_ics_tf)
        self.u_t_ics_pred = self.net_u_t(self.t_ics_tf, self.x_ics_tf)
        self.u_bc1_pred = self.net_u(self.t_bc1_tf, self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.t_bc2_tf, self.x_bc2_tf)

        self.u_pred = self.net_u(self.t_u_tf, self.x_u_tf)
        self.r_pred = self.net_r(self.t_r_tf, self.x_r_tf)
        
        self.u_ntk_pred = self.net_u(self.t_u_ntk_tf, self.x_u_ntk_tf)
        self.ut_ntk_pred = self.net_u_t(self.t_ut_ntk_tf, self.x_ut_ntk_tf)
        self.r_ntk_pred = self.net_r(self.t_r_ntk_tf, self.x_r_ntk_tf)

        # Boundary loss and Initial loss
        self.loss_ics_u = tf.reduce_mean(tf.square(self.u_ics_tf - self.u_ics_pred))
        self.loss_ics_u_t = tf.reduce_mean(tf.square(self.u_t_ics_pred))
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred))

        self.loss_bcs = self.loss_ics_u + self.loss_bc1 + self.loss_bc2

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_pred))

        # Total loss
        self.loss = self.lambda_r_tf * self.loss_res + self.lambda_u_tf * self.loss_bcs + self.lambda_ut_tf * self.loss_ics_u_t 

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Compute the Jacobian for weights and biases in each hidden layer  
        self.J_u = self.compute_jacobian(self.u_ntk_pred)
        self.J_ut = self.compute_jacobian(self.ut_ntk_pred)
        self.J_r = self.compute_jacobian(self.r_ntk_pred)
        
        self.K_u = self.compute_ntk(self.J_u, D1, self.J_u, D1)
        self.K_ut = self.compute_ntk(self.J_ut, D2, self.J_ut, D2)
        self.K_r = self.compute_ntk(self.J_r, D3, self.J_r, D3)

        # Loss logger
        self.loss_bcs_log = []
        self.loss_ut_ics_log = []
        self.loss_res_log = []
        self.l2_error_log = []

        # NTK logger
        self.K_u_log = []
        self.K_ut_log = []
        self.K_r_log = []
        
        # weights logger
        self.lambda_u_log = []
        self.lambda_ut_log = []
        self.lambda_r_log = []
        
         # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Saver
        self.saver = tf.train.Saver()

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                               dtype=tf.float32)

        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Forward pass for u
    def net_u(self, t, x):
        u = self.forward_pass(tf.concat([t, x], 1),
                              self.layers,
                              self.weights,
                              self.biases)
        return u

    def net_u_t(self, t, x):
        u_t = tf.gradients(self.net_u(t, x), t)[0] / self.sigma_t
        return u_t

    # Forward pass for f
    def net_r(self, t, x):
        u = self.net_u(t, x)
        residual = self.operator(u, t, x,
                                 self.c,
                                 self.sigma_t,
                                 self.sigma_x)
        return residual

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
    def compute_ntk(self, J1_list, D1, J2_list, D2):

        N = len(J1_list)
        
        Ker = tf.zeros((D1,D2))
        for k in range(N):
            J1 = tf.reshape(J1_list[k], shape=(D1,-1))
            J2 = tf.reshape(J2_list[k], shape=(D2,-1))
            
            K = tf.matmul(J1, tf.transpose(J2))
            Ker = Ker + K
        return Ker

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

        # Trains the model by minimizing the MSE loss

    def train(self, nIter=10000, batch_size=128, log_NTK=False, update_weights=False):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size // 3)
            X_bc1_batch, _ = self.fetch_minibatch(self.bcs_sampler[0], batch_size // 3)
            X_bc2_batch, _ = self.fetch_minibatch(self.bcs_sampler[1], batch_size // 3)
            
            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.t_ics_tf: X_ics_batch[:, 0:1], self.x_ics_tf: X_ics_batch[:, 1:2],
                       self.u_ics_tf: u_ics_batch,
                       self.t_bc1_tf: X_bc1_batch[:, 0:1], self.x_bc1_tf: X_bc1_batch[:, 1:2],
                       self.t_bc2_tf: X_bc2_batch[:, 0:1], self.x_bc2_tf: X_bc2_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 0:1], self.x_r_tf: X_res_batch[:, 1:2],
                       self.lambda_u_tf: self.lambda_u_val,
                       self.lambda_ut_tf: self.lambda_ut_val,
                       self.lambda_r_tf: self.lambda_r_val}

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time

                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value = self.sess.run(self.loss_bcs, tf_dict)
                loss_ics_ut_value = self.sess.run(self.loss_ics_u_t, tf_dict)
                loss_res_value = self.sess.run(self.loss_res, tf_dict)

                u_pred = self.predict_u(self.X_star)
                error = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)

                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_ut_ics_log.append(loss_ics_ut_value)
                self.l2_error_log.append(error)

                print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e,, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_bcs_value, loss_ics_ut_value, elapsed))

                print('lambda_u: {}'.format(self.lambda_u_val))
                print('lambda_ut: {}'.format(self.lambda_ut_val))
                print('lambda_r: {}'.format(self.lambda_r_val))

                start_time = timeit.default_timer()
            
            if log_NTK:
                X_bc_batch = np.vstack([X_ics_batch, X_bc1_batch, X_bc2_batch])
                X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size )
                
                if it % 1000 == 0:
                        print("Compute NTK...")
                        tf_dict = {self.t_u_ntk_tf: X_bc_batch[:,0 :1], self.x_u_ntk_tf: X_bc_batch[:, 1:2],
                                   self.t_ut_ntk_tf: X_ics_batch[:, 0:1], self.x_ut_ntk_tf: X_ics_batch[:, 1:2],
                                   self.t_r_ntk_tf: X_res_batch[:, 0:1], self.x_r_ntk_tf: X_res_batch[:, 1:2]}

                        # Compute NTK
                        K_u_value, K_ut_value, K_r_value = self.sess.run([self.K_u, self.K_ut, self.K_r], tf_dict)
                        
                        lambda_K_sum = np.trace(K_u_value) + np.trace(K_ut_value) + \
                                       np.trace(K_r_value)

                        # Store NTK and weights
                        self.K_u_log.append(K_u_value)
                        self.K_ut_log.append(K_ut_value)
                        self.K_r_log.append(K_r_value)

                        if update_weights:
                            self.lambda_u_val = lambda_K_sum / np.trace(K_u_value)
                            self.lambda_ut_val = lambda_K_sum /np.trace(K_ut_value)
                            self.lambda_r_val = lambda_K_sum / np.trace(K_r_value)

                        # Store weights
                        self.lambda_u_log.append(self.lambda_u_val)
                        self.lambda_ut_log.append(self.lambda_ut_val)
                        self.lambda_r_log.append(self.lambda_r_val)
          
    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1], self.x_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_r_tf: X_star[:, 0:1], self.x_r_tf: X_star[:, 1:2]}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star


class Wave1D_NTK_mFF:
    # Multiscale Fourier network with NTK adaptive weights

    # Initialize the class
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, c, kernel_size, X_star, u_star):
        # Normalize input
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # Test data
        self.X_star = X_star
        self.u_star = u_star

        # Initialize multi-scale Fourier features
        self.W1 = tf.Variable(tf.random_normal([2, layers[0] // 2], dtype=tf.float32) * 1.0,
                               dtype=tf.float32, trainable=True)

        self.W2 = tf.Variable(tf.random_normal([2, layers[0] // 2], dtype=tf.float32) * 10.0,
                               dtype=tf.float32, trainable=True)

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize weights for losses
        self.lambda_u_val = np.array(1.0)
        self.lambda_ut_val = np.array(1.0)
        self.lambda_r_val = np.array(1.0)

        # Wave velocity constant
        self.c = tf.constant(c, dtype=tf.float32)

        # Size of the NTK
        self.kernel_size = kernel_size

        D1 = self.kernel_size    # size of K_u
        D2 = self.kernel_size    # size of K_ut
        D3 = self.kernel_size    # size of K_r

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.lambda_u_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)
        self.lambda_ut_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)
        self.lambda_r_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)

        self.t_u_ntk_tf = tf.placeholder(tf.float32, shape=(D1, 1))
        self.x_u_ntk_tf = tf.placeholder(tf.float32, shape=(D1, 1))

        self.t_ut_ntk_tf = tf.placeholder(tf.float32, shape=(D2, 1))
        self.x_ut_ntk_tf = tf.placeholder(tf.float32, shape=(D2, 1))

        self.t_r_ntk_tf = tf.placeholder(tf.float32, shape=(D3, 1))
        self.x_r_ntk_tf = tf.placeholder(tf.float32, shape=(D3, 1))

        # Evaluate predictions
        self.u_ics_pred = self.net_u(self.t_ics_tf, self.x_ics_tf)
        self.u_t_ics_pred = self.net_u_t(self.t_ics_tf, self.x_ics_tf)
        self.u_bc1_pred = self.net_u(self.t_bc1_tf, self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.t_bc2_tf, self.x_bc2_tf)

        self.u_pred = self.net_u(self.t_u_tf, self.x_u_tf)
        self.r_pred = self.net_r(self.t_r_tf, self.x_r_tf)

        self.u_ntk_pred = self.net_u(self.t_u_ntk_tf, self.x_u_ntk_tf)
        self.ut_ntk_pred = self.net_u_t(self.t_ut_ntk_tf, self.x_ut_ntk_tf)
        self.r_ntk_pred = self.net_r(self.t_r_ntk_tf, self.x_r_ntk_tf)

        # Boundary loss and Initial loss
        self.loss_ics_u = tf.reduce_mean(tf.square(self.u_ics_tf - self.u_ics_pred))
        self.loss_ics_u_t = tf.reduce_mean(tf.square(self.u_t_ics_pred))
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred))

        self.loss_bcs = self.loss_ics_u + self.loss_bc1 + self.loss_bc2

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_pred))

        # Total loss
        self.loss = self.lambda_r_tf * self.loss_res + self.lambda_u_tf * self.loss_bcs + self.lambda_ut_tf * self.loss_ics_u_t

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Compute the Jacobian for weights and biases in each hidden layer
        self.J_u = self.compute_jacobian(self.u_ntk_pred)
        self.J_ut = self.compute_jacobian(self.ut_ntk_pred)
        self.J_r = self.compute_jacobian(self.r_ntk_pred)

        self.K_u = self.compute_ntk(self.J_u, D1, self.J_u, D1)
        self.K_ut = self.compute_ntk(self.J_ut, D2, self.J_ut, D2)
        self.K_r = self.compute_ntk(self.J_r, D3, self.J_r, D3)

        # Loss logger
        self.loss_bcs_log = []
        self.loss_ut_ics_log = []
        self.loss_res_log = []
        self.l2_error_log = []

        # NTK logger
        self.K_u_log = []
        self.K_ut_log = []
        self.K_r_log = []

        # weights logger
        self.lambda_u_log = []
        self.lambda_ut_log = []
        self.lambda_r_log = []

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Saver
        self.saver = tf.train.Saver()

    # Initialize network weights and biases using Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

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
        # W = self.xavier_init(size=[layers[-2], layers[-1]])
        b = tf.Variable(tf.random_normal([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)

        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)

        # Multi-scale Fourier feature encodings
        H1 = tf.concat([tf.sin(tf.matmul(H, self.W1)),
                        tf.cos(tf.matmul(H, self.W1))], 1)
        H2 = tf.concat([tf.sin(tf.matmul(H, self.W2)),
                        tf.cos(tf.matmul(H, self.W2))], 1)

        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]

            H1 = tf.tanh(tf.add(tf.matmul(H1, W), b))
            H2 = tf.tanh(tf.add(tf.matmul(H2, W), b))

        # Merge the outputs by concatenation
        H = tf.concat([H1, H2], 1)

        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)

        return H

    # Forward pass for u
    def net_u(self, t, x):
        u = self.forward_pass(tf.concat([t, x], 1))
        return u

    def net_u_t(self, t, x):
        u_t = tf.gradients(self.net_u(t, x), t)[0] / self.sigma_t
        return u_t

    # Forward pass for f
    def net_r(self, t, x):
        u = self.net_u(t, x)
        residual = self.operator(u, t, x,
                                 self.c,
                                 self.sigma_t,
                                 self.sigma_x)
        return residual

    # Compute Jacobian for each weights and biases in each layer and retrun a list
    def compute_jacobian(self, f):
        J_list = []
        L = len(self.weights)
        for i in range(L):
            J_w = jacobian(f, self.weights[i])
            J_list.append(J_w)

        for i in range(L):
            J_b = jacobian(f, self.biases[i])
            J_list.append(J_b)
        return J_list

    # Compute the empirical NTK = J J^T
    def compute_ntk(self, J1_list, D1, J2_list, D2):

        N = len(J1_list)

        Ker = tf.zeros((D1, D2))
        for k in range(N):
            J1 = tf.reshape(J1_list[k], shape=(D1, -1))
            J2 = tf.reshape(J2_list[k], shape=(D2, -1))

            K = tf.matmul(J1, tf.transpose(J2))
            Ker = Ker + K
        return Ker

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128, log_NTK=False, update_weights=False):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size // 3)
            X_bc1_batch, _ = self.fetch_minibatch(self.bcs_sampler[0], batch_size // 3)
            X_bc2_batch, _ = self.fetch_minibatch(self.bcs_sampler[1], batch_size // 3)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.t_ics_tf: X_ics_batch[:, 0:1], self.x_ics_tf: X_ics_batch[:, 1:2],
                       self.u_ics_tf: u_ics_batch,
                       self.t_bc1_tf: X_bc1_batch[:, 0:1], self.x_bc1_tf: X_bc1_batch[:, 1:2],
                       self.t_bc2_tf: X_bc2_batch[:, 0:1], self.x_bc2_tf: X_bc2_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 0:1], self.x_r_tf: X_res_batch[:, 1:2],
                       self.lambda_u_tf: self.lambda_u_val,
                       self.lambda_ut_tf: self.lambda_ut_val,
                       self.lambda_r_tf: self.lambda_r_val}

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time

                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value = self.sess.run(self.loss_bcs, tf_dict)
                loss_ics_ut_value = self.sess.run(self.loss_ics_u_t, tf_dict)
                loss_res_value = self.sess.run(self.loss_res, tf_dict)

                u_pred = self.predict_u(self.X_star)
                error = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)

                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_ut_ics_log.append(loss_ics_ut_value)
                self.l2_error_log.append(error)

                print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e,, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_bcs_value, loss_ics_ut_value, elapsed))

                print('lambda_u: {}'.format(self.lambda_u_val))
                print('lambda_ut: {}'.format(self.lambda_ut_val))
                print('lambda_r: {}'.format(self.lambda_r_val))

                start_time = timeit.default_timer()

            if log_NTK:
                X_bc_batch = np.vstack([X_ics_batch, X_bc1_batch, X_bc2_batch])
                X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size)

                if it % 100 == 0:
                    print("Compute NTK...")
                    tf_dict = {self.t_u_ntk_tf: X_bc_batch[:, 0:1], self.x_u_ntk_tf: X_bc_batch[:, 1:2],
                               self.t_ut_ntk_tf: X_ics_batch[:, 0:1], self.x_ut_ntk_tf: X_ics_batch[:, 1:2],
                               self.t_r_ntk_tf: X_res_batch[:, 0:1], self.x_r_ntk_tf: X_res_batch[:, 1:2]}

                    K_u_value, K_ut_value, K_r_value = self.sess.run([self.K_u, self.K_ut, self.K_r], tf_dict)

                    # Store NTK
                    self.K_u_log.append(K_u_value)
                    self.K_ut_log.append(K_ut_value)
                    self.K_r_log.append(K_r_value)

                    if update_weights:
                        lambda_K_sum = np.trace(K_u_value) + np.trace(K_ut_value) + \
                                       np.trace(K_r_value)

                        # Update weights
                        self.lambda_u_val = lambda_K_sum / np.trace(K_u_value)
                        self.lambda_ut_val = lambda_K_sum / np.trace(K_ut_value)
                        self.lambda_r_val = lambda_K_sum / np.trace(K_r_value)

                    # Store weights
                    self.lambda_u_log.append(self.lambda_u_val)
                    self.lambda_ut_log.append(self.lambda_ut_val)
                    self.lambda_r_log.append(self.lambda_r_val)

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1], self.x_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_r_tf: X_star[:, 0:1], self.x_r_tf: X_star[:, 1:2]}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star


class Wave1D_NTK_ST_mFF:
    # Initialize the class
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, c, kernel_size, X_star, u_star):
        # Normalization constants
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # Test data
        self.X_star = X_star
        self.u_star = u_star

        # Initialize spatial and temporal Fourier features
        self.W1_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 1.0,
                               dtype=tf.float32, trainable=False)

        self.W2_t = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 10.0,
                               dtype=tf.float32, trainable=False)

        self.W1_x = tf.Variable(tf.random_normal([1, layers[0] // 2], dtype=tf.float32) * 1.0,
                               dtype=tf.float32, trainable=False)

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # weights
        self.lambda_u_val = np.array(1.0)
        self.lambda_ut_val = np.array(1.0)
        self.lambda_r_val = np.array(1.0)

        # Wave velocity constant
        self.c = tf.constant(c, dtype=tf.float32)

        # Size of NTK
        self.kernel_size = kernel_size

        D1 = self.kernel_size    # size of K_u
        D2 = self.kernel_size    # size of K_ut
        D3 = self.kernel_size    # size of K_r

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_ics_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc1_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_bc2_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.lambda_u_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)
        self.lambda_ut_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)
        self.lambda_r_tf = tf.placeholder(tf.float32, shape=self.lambda_u_val.shape)

        self.t_u_ntk_tf = tf.placeholder(tf.float32, shape=(D1, 1))
        self.x_u_ntk_tf = tf.placeholder(tf.float32, shape=(D1, 1))

        self.t_ut_ntk_tf = tf.placeholder(tf.float32, shape=(D2, 1))
        self.x_ut_ntk_tf = tf.placeholder(tf.float32, shape=(D2, 1))

        self.t_r_ntk_tf = tf.placeholder(tf.float32, shape=(D3, 1))
        self.x_r_ntk_tf = tf.placeholder(tf.float32, shape=(D3, 1))

        # Evaluate predictions
        self.u_ics_pred = self.net_u(self.t_ics_tf, self.x_ics_tf)
        self.u_t_ics_pred = self.net_u_t(self.t_ics_tf, self.x_ics_tf)
        self.u_bc1_pred = self.net_u(self.t_bc1_tf, self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.t_bc2_tf, self.x_bc2_tf)

        self.u_pred = self.net_u(self.t_u_tf, self.x_u_tf)
        self.r_pred = self.net_r(self.t_r_tf, self.x_r_tf)

        self.u_ntk_pred = self.net_u(self.t_u_ntk_tf, self.x_u_ntk_tf)
        self.ut_ntk_pred = self.net_u_t(self.t_ut_ntk_tf, self.x_ut_ntk_tf)
        self.r_ntk_pred = self.net_r(self.t_r_ntk_tf, self.x_r_ntk_tf)

        # Boundary loss and Initial loss
        self.loss_ics_u = tf.reduce_mean(tf.square(self.u_ics_tf - self.u_ics_pred))
        self.loss_ics_u_t = tf.reduce_mean(tf.square(self.u_t_ics_pred))
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_pred))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_pred))

        self.loss_bcs = self.loss_ics_u + self.loss_bc1 + self.loss_bc2

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_pred))

        # Total loss
        self.loss = self.lambda_r_tf * self.loss_res + self.lambda_u_tf * self.loss_bcs + self.lambda_ut_tf * self.loss_ics_u_t

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        # Compute the Jacobian for weights and biases in each hidden layer
        self.J_u = self.compute_jacobian(self.u_ntk_pred)
        self.J_ut = self.compute_jacobian(self.ut_ntk_pred)
        self.J_r = self.compute_jacobian(self.r_ntk_pred)

        self.K_u = self.compute_ntk(self.J_u, D1, self.J_u, D1)
        self.K_ut = self.compute_ntk(self.J_ut, D2, self.J_ut, D2)
        self.K_r = self.compute_ntk(self.J_r, D3, self.J_r, D3)

        # Loss logger
        self.loss_bcs_log = []
        self.loss_ut_ics_log = []
        self.loss_res_log = []
        self.l2_error_log = []

        # NTK logger
        self.K_u_log = []
        self.K_ut_log = []
        self.K_r_log = []

        # weights logger
        self.lambda_u_log = []
        self.lambda_ut_log = []
        self.lambda_r_log = []

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Saver
        self.saver = tf.train.Saver()

    # Initialize network weights and biases using Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

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

    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)

        t = H[:, 0:1]
        x = H[:, 1:2]

        # Temporal and spatial Fourier feature encodings
        H1_t = tf.concat([tf.sin(tf.matmul(t, self.W1_t)),
                         tf.cos(tf.matmul(t, self.W1_t))], 1)

        H2_t = tf.concat([tf.sin(tf.matmul(t, self.W2_t)),
                          tf.cos(tf.matmul(t, self.W2_t))], 1)

        H1_x = tf.concat([tf.sin(tf.matmul(x, self.W1_x)),
                          tf.cos(tf.matmul(x, self.W1_x))], 1)

        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]

            H1_t = tf.tanh(tf.add(tf.matmul(H1_t, W), b))
            H2_t = tf.tanh(tf.add(tf.matmul(H2_t, W), b))
            H1_x = tf.tanh(tf.add(tf.matmul(H1_x, W), b))

        # Merge outputs
        H1 = tf.multiply(H1_t, H1_x)
        H2 = tf.multiply(H2_t, H1_x)
        H = tf.concat([H1, H2], 1)

        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)

        return H

    # Forward pass for u
    def net_u(self, t, x):
        u = self.forward_pass(tf.concat([t, x], 1))
        return u

    def net_u_t(self, t, x):
        u_t = tf.gradients(self.net_u(t, x), t)[0] / self.sigma_t
        return u_t

    # Forward pass for f
    def net_r(self, t, x):
        u = self.net_u(t, x)
        residual = self.operator(u, t, x,
                                 self.c,
                                 self.sigma_t,
                                 self.sigma_x)
        return residual

    # Compute Jacobian for each weights and biases in each layer and retrun a list
    def compute_jacobian(self, f):
        J_list = []
        L = len(self.weights)
        for i in range(L):
            J_w = jacobian(f, self.weights[i])
            J_list.append(J_w)

        for i in range(L):
            J_b = jacobian(f, self.biases[i])
            J_list.append(J_b)
        return J_list

    # Compute the empirical NTK = J J^T
    def compute_ntk(self, J1_list, D1, J2_list, D2):

        N = len(J1_list)

        Ker = tf.zeros((D1, D2))
        for k in range(N):
            J1 = tf.reshape(J1_list[k], shape=(D1, -1))
            J2 = tf.reshape(J2_list[k], shape=(D2, -1))

            K = tf.matmul(J1, tf.transpose(J2))
            Ker = Ker + K
        return Ker

    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=128, log_NTK=False, update_weights=False):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size // 3)
            X_bc1_batch, _ = self.fetch_minibatch(self.bcs_sampler[0], batch_size // 3)
            X_bc2_batch, _ = self.fetch_minibatch(self.bcs_sampler[1], batch_size // 3)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.t_ics_tf: X_ics_batch[:, 0:1], self.x_ics_tf: X_ics_batch[:, 1:2],
                       self.u_ics_tf: u_ics_batch,
                       self.t_bc1_tf: X_bc1_batch[:, 0:1], self.x_bc1_tf: X_bc1_batch[:, 1:2],
                       self.t_bc2_tf: X_bc2_batch[:, 0:1], self.x_bc2_tf: X_bc2_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 0:1], self.x_r_tf: X_res_batch[:, 1:2],
                       self.lambda_u_tf: self.lambda_u_val,
                       self.lambda_ut_tf: self.lambda_ut_val,
                       self.lambda_r_tf: self.lambda_r_val}

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time

                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value = self.sess.run(self.loss_bcs, tf_dict)
                loss_ics_ut_value = self.sess.run(self.loss_ics_u_t, tf_dict)
                loss_res_value = self.sess.run(self.loss_res, tf_dict)
                
                u_pred = self.predict_u(self.X_star)
                error = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)

                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_ut_ics_log.append(loss_ics_ut_value)
                self.l2_error_log.append(error)

                print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e,, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_bcs_value, loss_ics_ut_value, elapsed))

                print('lambda_u: {}'.format(self.lambda_u_val))
                print('lambda_ut: {}'.format(self.lambda_ut_val))
                print('lambda_r: {}'.format(self.lambda_r_val))

                start_time = timeit.default_timer()

            if log_NTK:
                X_bc_batch = np.vstack([X_ics_batch, X_bc1_batch, X_bc2_batch])
                X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size)

                if it % 100 == 0:
                    print("Compute NTK...")
                    tf_dict = {self.t_u_ntk_tf: X_bc_batch[:, 0:1], self.x_u_ntk_tf: X_bc_batch[:, 1:2],
                               self.t_ut_ntk_tf: X_ics_batch[:, 0:1], self.x_ut_ntk_tf: X_ics_batch[:, 1:2],
                               self.t_r_ntk_tf: X_res_batch[:, 0:1], self.x_r_ntk_tf: X_res_batch[:, 1:2]}

                    K_u_value, K_ut_value, K_r_value = self.sess.run([self.K_u, self.K_ut, self.K_r], tf_dict)


                    self.K_u_log.append(K_u_value)
                    self.K_ut_log.append(K_ut_value)
                    self.K_r_log.append(K_r_value)

                    if update_weights:
                        lambda_K_sum = np.trace(K_u_value) + np.trace(K_ut_value) + \
                                       np.trace(K_r_value)

                        self.lambda_u_val = lambda_K_sum / np.trace(K_u_value)
                        self.lambda_ut_val = lambda_K_sum / np.trace(K_ut_value)
                        self.lambda_r_val = lambda_K_sum / np.trace(K_r_value)

                    # Store weights
                    self.lambda_u_log.append(self.lambda_u_val)
                    self.lambda_ut_log.append(self.lambda_ut_val)
                    self.lambda_r_log.append(self.lambda_r_val)

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1], self.x_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_r_tf: X_star[:, 0:1], self.x_r_tf: X_star[:, 1:2]}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star



    


