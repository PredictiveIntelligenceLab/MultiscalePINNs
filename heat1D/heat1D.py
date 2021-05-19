import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from models_tf import Sampler, heat1D_NN, heat1D_FF, heat1D_ST_FF

if __name__ == '__main__':

    # Define exact solution
    def u(x, a, b):
        """
        :param x: x = (t, x)
        """

        t  = x[:,0:1]
        x = x[:,1:2]
        
        return np.exp(-a * t) * np.sin(b * np.pi * x)

    def u_t(x, a, b):
        return - a * u(x, a, b)

    def u_xx(x, a, b):
        return - (b * np.pi)**2 * u(x, a, b)

    def f(x, a, b):
        k = a / (b * np.pi)**2 
        return u_t(x, a, b) - k * u_xx(x, a, b)

    # Define PDE residual
    def operator(u, t, x, k,  sigma_t=1.0, sigma_x=1.0):
        u_t = tf.gradients(u, t)[0] / sigma_t
        u_x = tf.gradients(u, x)[0] / sigma_x
        u_xx = tf.gradients(u_x, x)[0] / sigma_x
        residual = u_t - k * u_xx
        return residual

    # Parameters of equations
    a = 1
    b = 500
    k = a / (b * np.pi)**2 

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
    ics_sampler = Sampler(2, ics_coords, lambda x: u(x, a, b), name='Initial Condition 1')

    # Create boundary conditions samplers
    bc1 = Sampler(2, bc1_coords, lambda x: u(x, a, b), name='Dirichlet BC1')
    bc2 = Sampler(2, bc2_coords, lambda x: u(x, a, b), name='Dirichlet BC2')
    bcs_sampler = [bc1, bc2]

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, lambda x: f(x, a, b), name='Forcing')

    # Test data
    nn = 100  # nn = 1000
    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

    u_star = u(X_star, a, b)
    f_star = f(X_star, a, b)

    # Define model
    # heat1D_NN: Plain MLP
    # heat1D_FF: Plain Fourier feature network
    # heat1D_ST_FF: Spatial-temporal Plain Fourier feature network
    
    layers = [100, 100, 100, 1]  # For heat1D_NN, use layers = [1, 100, 100, 100, 1]
    sigma = 500   # Hyper-parameter for Fourier feature embeddings
    model = heat1D_NN(layers, operator, k, 
                             ics_sampler, bcs_sampler, res_sampler, 
                             sigma, X_star, u_star)

    # Train model
    model.train(nIter=40000, batch_size=128)


    # Predictions
    u_pred = model.predict_u(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Relative L2 error_u: {:.2e}'.format(error_u))
    

    # Grid data
    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
    F_star = griddata(X_star, f_star.flatten(), (t, x), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
    
    
    # Plot
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(t, x, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(r'Exact')
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(t, x, U_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(r'Predicted')
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()

    loss_ics = model.loss_ics_log
    loss_bcs = model.loss_bcs_log
    loss_res = model.loss_res_log
    l2_error = model.l2_error_log
    
    fig_2 = plt.figure(2, figsize=(6, 5))
    with sns.axes_style("darkgrid"):
        iters = 100 * np.arange(len(loss_res))
            
        plt.plot(iters, loss_res, label='$\mathcal{L}_{r}$', linewidth=2)
        plt.plot(iters, loss_bcs, label='$\mathcal{L}_{bc}$', linewidth=2)
        plt.plot(iters, loss_ics, label='$\mathcal{L}_{ic}$', linewidth=2)
        plt.plot(iters, l2_error, label=r'$L^2$ error', linewidth=2)
        
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2, fontsize=17)
        plt.tight_layout()
        plt.show()

        
    
