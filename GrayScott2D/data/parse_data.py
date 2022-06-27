# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:10:47 2020

@author: Wsf12
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

data = sio.loadmat('sol.mat')

X = data['X']
Y = data['Y']

tspan = data['tspan'].flatten()

usol =  data['usol']
vsol =  data['vsol']

epsilon1 = data['ep1']
epsilon2 = data['ep2']
b = data['b']
d = data['d']

X_list = []
U_list = []

steps = len(tspan) # 500 snap shots

x = X.flatten()[:,None]
y = Y.flatten()[:,None]

T1 = 350
T2 = 400   

for k in range(T1, T2 + 1):
    t = tspan[k] * np.ones_like(x) 
    
    u = usol[:,:,k].flatten()[:, None]
    v = vsol[:,:,k].flatten()[:, None]
    
    X_list.append(np.concatenate([t, x, y], axis = 1))
    U_list.append(np.concatenate([u, v], axis = 1))

X = np.vstack(X_list)
U = np.vstack(U_list)

data_dict = {'X': X, 'U': U, 
             'tspan': tspan, 'T1': tspan[T1], 'T2': tspan[T2],  
             'ep1':epsilon1, 'ep2':epsilon2, 'b':b, 'd':d}

np.save('data.npy', data_dict)


##  data down sampling
#X_reduced = data['X_reduced']
#Y_reduced = data['Y_reduced']
#
#tspan = data['tspan'].flatten()
#
#usol =  data['usol_reduced']
#vsol =  data['vsol_reduced']
#
#epsilon1 = data['ep1']
#epsilon2 = data['ep2']
#b = data['b']
#d = data['d']
#
#X_list = []
#U_list = []
#
#steps = len(tspan) # 500 snap shots
#
#x = X_reduced.flatten()[:,None]
#y = Y_reduced.flatten()[:,None]
#
#for k in range(T1, T2):
#    t = tspan[k] * np.ones_like(x) 
#    
#    u = usol[:,:,k].flatten()[:, None]
#    v = vsol[:,:,k].flatten()[:, None]
#    
#    X_list.append(np.concatenate([t, x, y], axis = 1))
#    U_list.append(np.concatenate([u, v], axis = 1))
#
#X_reduced = np.vstack(X_list)
#U_reduced = np.vstack(U_list)
#
#data_dict = {'X_reduced': X_reduced, 'U_reduced': U_reduced, 
#             'tspan': tspan, 'T1': tspan[T1], 'T2': tspan[T2], 
#             'ep1':epsilon1, 'ep2':epsilon2, 'b':b, 'd':d}
#
#np.save('data_reduced.npy', data_dict)


