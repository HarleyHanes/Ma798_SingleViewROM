#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Aa798_SingelViewROA/python/test_rom.py
@Time    :   2023/11/20 17:59:17
@Author  :   Harley Hanes 
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2023, Harley Hanes
@Desc    :   None
'''


import numpy as np
import fluid_rom
import plots
import data
from rom_updating import *


location = "Ma798_SingleViewROM/Fiber_Data/"

t_start=100
n_initial_snapshots = 50
n_modes = 4
iterations =6

x = np.loadtxt(location+'xs.dat')
print("x.shape: ", x.shape)
times = np.loadtxt(location+'times.dat')[t_start:]
print("times.shape: ", times.shape)
snapshots_full = np.loadtxt(location+'data.dat')[:,t_start:]
print("snapshots_full.shape: ", snapshots_full.shape)
averages = np.mean(snapshots_full, axis =1)

averages=np.reshape(averages,(averages.size,1))

snapshots_full=snapshots_full-averages

snapshots = snapshots_full[:,0:n_initial_snapshots]


(error_SVD, snapshots_SVD) = IterativeOptimization(snapshots, snapshots_full, averages, n_modes, times, x[1]-x[0], iterations = iterations, method="SVD")
(error_RandSVD, snapshots_RandSVD) = IterativeOptimization(snapshots, snapshots_full, averages, n_modes, times, x[1]-x[0], iterations = iterations, method="RandSVD")
#(error_SingleView, snapshots_SingleView) = IterativeOptimization(snapshots, snapshots_full, averages, n_modes, times, x[1]-x[0], iterations = iterations, method="SingleView")
error_plot = np.sum(np.abs(error_RandSVD),axis=1)/np.sum(np.abs(error_SVD),axis = 1)
#error_plot = np.array([np.sum(error_SVD,axis = 1), np.sum(error_RandSVD,axis = 1)]).transpose()
plots.plot_method_error(error_plot, np.arange(iterations), xlabel = "Greedy Sampling Iterations", ylabel = "$\\frac{Error_{RandSVD}}{Error_{SVD}}$", 
                        save_path = "Ma798_SingleViewROM/figures/GreedySampling/convergence")

