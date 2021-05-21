import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from wave_models_tf import Sampler, Wave1D_NTK, Wave1D_NTK_mFF, Wave1D_NTK_ST_mFF

if __name__ == '__main__':
    def u(x, a, c):
        """
        :param x: x = (t, x)
        """
        t = x[:,0:1]
        x = x[:,1:2]
        return np.sin(np.pi * x) * np.cos(c * np.pi * t) + \
                np.sin(a * np.pi* x) * np.cos(a * c  * np.pi * t)

    def f(x, a, c):
        N = x.shape[0]
        return  np.zeros((N,1))

    def operator(u, t, x, c, sigma_t=1.0, sigma_x=1.0):
        u_t = tf.gradients(u, t)[0] / sigma_t
        u_x = tf.gradients(u, x)[0] / sigma_x
        u_tt = tf.gradients(u_t, t)[0] / sigma_t
        u_xx = tf.gradients(u_x, x)[0] / sigma_x
        residual = u_tt - c**2 * u_xx
        return residual
    
    # Hyper-parameters
    a = 2
    c = 10
    
    # Domain boundaries
    ics_coords = np.array([[0.0, 0.0],
                           [0.0, 1.0]])
    bc1_coords = np.array([[0.0, 0.0],
                           [1.0, 0.0]])
    bc2_coords = np.array([[0.0, 1.0],
                           [1.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    # Create initial conditions samplers
    ics_sampler = Sampler(2, ics_coords, lambda x: u(x, a, c), name='Initial Condition 1')

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: u(x, a, c), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: u(x, a, c), name='Dirichlet BC2')
    bcs_sampler = [bc1, bc2]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: f(x, a, c), name='Forcing')
    
    # Test data
    nn = 200
    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

    u_star = u(X_star, a,c)

    # Define model
    # Wave1D_NTK: Plain MLP with NTK adaptive weights
    # Wave1D_NTK_mFF: Multi-scale Fourier feature network with NTK adaptive weights
    # Wave1D_NTK_ST_mFF: Spatial-temporal Fourier feature network with NTK adaptive weights
    
#    layers = [2, 200, 200, 200, 1]    # if use Wave1D_NTK model
    layers = [200, 200, 200, 1]

    kernel_size = 120
    model = Wave1D_NTK_mFF(layers, operator, ics_sampler, bcs_sampler, res_sampler, c, kernel_size, X_star, u_star)
    
    
    # Train model
    itertaions = 40001
    model.train(nIter=itertaions, batch_size =kernel_size, log_NTK=True, update_weights=True)

    # Predictions
    u_pred = model.predict_u(X_star)
    f_pred = model.predict_r(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    print('Relative L2 error_u: %e' % (error_u))
    
    # Plot
    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
    
    # Predictions    
    fig = plt.figure(3, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(t, x, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Exact u(x)')

    plt.subplot(1, 3, 2)
    plt.pcolor(t, x, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Predicted u(x)')

    plt.subplot(1, 3, 3)
    plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()
    
    # Restore loss_res and loss_bcs
    loss_res = model.loss_res_log
    loss_bcs = model.loss_bcs_log
    loss_u_t_ics = model.loss_ut_ics_log

    l2_error = model.l2_error_log

    fig = plt.figure(figsize=(6,5))
    iters =100 *  np.arange(len(loss_res))
    with sns.axes_style("darkgrid"):
        plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$')
        plt.plot(iters, loss_bcs, label='$\mathcal{L}_{u}$')
        plt.plot(iters, loss_u_t_ics, label='$\mathcal{L}_{u_t}$')
        plt.plot(iters, l2_error, label='$\mathcal{L}^2 error$')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.show()

    # NTK
    # Create loggers for eigenvalues of NTK
    lambda_K_u_log = []
    lambda_K_ut_log = []
    lambda_K_r_log = []
    
    # Restore the NTK
    K_u_list = model.K_u_log
    K_ut_list = model.K_ut_log
    K_r_list = model.K_r_log
        
    for k in range(len(K_u_list)):
        K_u = K_u_list[k]
        K_ut = K_ut_list[k]
        K_r = K_r_list[k]
            
        # Compute eigenvalues
        lambda_K_u, _ = np.linalg.eig(K_u)
        lambda_K_ut, _ = np.linalg.eig(K_ut)
        lambda_K_r, _ = np.linalg.eig(K_r)
        # Sort in descresing order
        lambda_K_u = np.sort(np.real(lambda_K_u))[::-1]
        lambda_K_ut = np.sort(np.real(lambda_K_ut))[::-1]
        lambda_K_r = np.sort(np.real(lambda_K_r))[::-1]
        
        # Store eigenvalues
        lambda_K_u_log.append(lambda_K_u)
        lambda_K_ut_log.append(lambda_K_ut)
        lambda_K_r_log.append(lambda_K_r)
    
    #     Eigenvalues of NTK
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1,3,1)
    plt.plot(lambda_K_u_log[0], label = '$n=0$')
    plt.plot(lambda_K_u_log[1], '--', label = '$n=10,000$')
    plt.plot(lambda_K_u_log[4], '--', label = '$n=40,000$')
    plt.plot(lambda_K_u_log[-1], '--', label = '$n=80,000$')
    plt.xlabel('index')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(r'Eigenvalues of ${K}_u$')

    plt.subplot(1,3,2)
    plt.plot(lambda_K_ut_log[0], label = '$n=0$')
    plt.plot(lambda_K_ut_log[1], '--',label = '$n=10,000$')
    plt.plot(lambda_K_ut_log[4], '--', label = '$n=40,000$')
    plt.plot(lambda_K_ut_log[-1], '--', label = '$n=80,000$')
    plt.xlabel('index')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(r'Eigenvalues of ${K}_{u_t}$')
    
    ax =plt.subplot(1,3,3)
    plt.plot(lambda_K_r_log[0], label = '$n=0$')
    plt.plot(lambda_K_r_log[1], '--', label = '$n=10,000$')
    plt.plot(lambda_K_r_log[4], '--', label = '$n=40,000$')
    plt.plot(lambda_K_r_log[-1], '--', label = '$n=80,000$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('index')
    plt.title(r'Eigenvalues of ${K}_{r}$')
    plt.legend()
    plt.tight_layout()
    plt.show()
     
    # Evolution of weights during training
    lambda_u_log = model.lambda_u_log
    lambda_ut_log = model.lambda_ut_log
    lambda_r_log = model.lambda_r_log   

    fig = plt.figure(figsize=(6, 5))
    plt.plot(lambda_u_log, label='$\lambda_u$')
    plt.plot(lambda_ut_log, label='$\lambda_{u_t}$')
    plt.plot(lambda_r_log, label='$\lambda_{r}$')
    plt.xlabel('iterations')
    plt.ylabel('$\lambda$')
    plt.yscale('log')
    plt.legend( )
    plt.locator_params(axis='x',nbins=5)
    plt.tight_layout()
    plt.show()   
    
