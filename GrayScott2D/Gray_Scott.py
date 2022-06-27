import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import griddata
from models_tf import Sampler, ResidualSampler, DataSampler, Gray_Scott2D

if __name__ == '__main__':
    # Reload  data
    datafile = 'data.npy'
    data = np.load(datafile, allow_pickle=True).item()

    X = data['X']
    U = data['U']

    # Time intervals
    tspan = data['tspan']
    T1 = data['T1']
    T2 = data['T2']

    # Parameters
    epsilon1 = data['ep1']
    epsilon2 = data['ep2']
    b = data['b']
    d = data['d']

    # Define data sampler and residual sampler
    dom_coords = np.array([[T1, -1.0, -1.0],
                           [T2, 1.0, 1.0]])
    res_sampler = Sampler(3, dom_coords, lambda x: np.zeros_like(x), name='Forcing')

    data_sampler = DataSampler(X, U)

    # Create model
    layers = [3, 100, 100, 100, 100, 100, 100, 100, 2]
    model = Gray_Scott2D(data_sampler, res_sampler, layers, b, d)

    # Train model
    model.train(nIter=120000, batch_size=1000)

    # Save results
    model.saver.save(model.sess, 'SavedModels/' 'GS_param' + '_7x100_it120000' + '.ckpt')

    ep1 = model.sess.run(model.epsilon1)
    ep2 = model.sess.run(model.epsilon2)

    print('ep1: {:.3e}, ep2: {:.3e}'.format(np.exp(ep1), np.exp(ep2)))

    ep1_log = model.ep1_log
    ep2_log = model.ep2_log

    np.savetxt('SavedResults/' + 'ep1_log_original', ep1_log, delimiter=',')
    np.savetxt('SavedResults/' + 'ep2_log_original', ep2_log, delimiter=',')

    # Prediction
    raw_data = sio.loadmat('sol.mat')

    X = raw_data['X']
    Y = raw_data['Y']
    tspan = raw_data['tspan'].flatten()
    usol = raw_data['usol']
    vsol = raw_data['vsol']

    step = -50
    x = X.flatten()[:, None]
    y = Y.flatten()[:, None]
    t = tspan[step] * np.ones_like(x)

    X_star = np.concatenate([t, x, y], axis=1)

    u_pred, v_pred = model.predict(X_star)
    u_star = usol[:, :, step].flatten()[:, None]
    v_star = vsol[:, :, step].flatten()[:, None]

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)

    print('Relative L2 error_u: %e' % (error_u))
    print('Relative L2 error_v: %e' % (error_v))

    ep1 = model.sess.run(model.epsilon1)
    ep2 = model.sess.run(model.epsilon2)

    print('ep1: {:.3e}, ep2: {:.3e}'.format(np.exp(ep1), np.exp(ep2)))

    # Plot
    U_star = griddata(np.concatenate([x, y], axis=1), u_pred.flatten(), (X, Y), method='cubic')
    V_star = griddata(np.concatenate([x, y], axis=1), v_pred.flatten(), (X, Y), method='cubic')

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(X, Y, U_star)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.pcolor(X, Y, usol[:, :, step])
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.pcolor(X, Y, U_star - usol[:, :, step])
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(X, Y, V_star)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.pcolor(X, Y, vsol[:, :, step])
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.pcolor(X, Y, V_star - vsol[:, :, step])
    plt.colorbar()
    plt.show()


