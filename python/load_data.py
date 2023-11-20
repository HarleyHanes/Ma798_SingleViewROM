#IMPORTS

import numpy as np
from POD import POD
from plots import *
import matplotlib.pyplot as plt

tstart = 0
tend = 201

xs = np.loadtxt('xs.dat')
times = np.loadtxt('times.dat')
data = np.loadtxt('data.dat')

xlen = data.shape[0]
time_len = data.shape[1]
        
dx = xs[1]-xs[0]
dt = times[1]-times[0]

L = xs[-1]+dx

data = data[:,tstart:tend]
times = times[tstart:tend]

# POD

k = 4 # number of modes

spatial, temporal = POD(data,k)

# Visualize Data

plot_solution(data,xs,times)
plt.figure(2)
plot_temporal(temporal,times,k)
plt.figure(3)
plot_spatial(spatial,xs,k)
plt.show()
