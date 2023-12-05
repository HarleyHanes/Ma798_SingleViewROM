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
n_initial_snapshots = 30
n_modes = 4
iterations =6
batchsize=3

x = np.loadtxt(location+'xs.dat')
print("x.shape: ", x.shape)
times = np.loadtxt(location+'times.dat')[t_start:]
times_initial = times[:n_initial_snapshots]
indices = np.arange(n_initial_snapshots)
print("times.shape: ", times.shape)
snapshots_full = np.loadtxt(location+'data.dat')[:,t_start:]
print("snapshots_full.shape: ", snapshots_full.shape)
averages = np.mean(snapshots_full, axis =1)

averages=np.reshape(averages,(averages.size,1))

#snapshots_full=snapshots_full-averages

snapshots = snapshots_full[:,0:n_initial_snapshots]


(error_SVD, snapshots_SVD) = IterativeOptimization(snapshots, snapshots_full, indices, n_modes, times_initial, times, x[1]-x[0], batchsize= batchsize, iterations = iterations, method="SVD")
(error_RandSVD, snapshots_RandSVD) = IterativeOptimization(snapshots, snapshots_full, indices, n_modes, times_initial,times, x[1]-x[0], batchsize= batchsize, iterations = iterations, method="RandSVD")
(error_SingleView, snapshots_SingleView) = IterativeOptimization(snapshots, snapshots_full, indices, n_modes, times_initial, times, x[1]-x[0], batchsize= batchsize, iterations = iterations, method="SingleView")
error = np.array([error_SVD, error_RandSVD, error_SingleView])
error_SSQ = np.sqrt(np.sum(error**2,axis=2))
print("error.shape: ", error.shape)
print("error_SSQ.shape: ", error_SSQ.shape)

error_ratios = error_SSQ[1:3,:]/error_SSQ[0,:]
print("error_ratios.shape: ", error_ratios.shape)
#error_plot = np.array([np.sum(error_SVD,axis = 1), np.sum(error_RandSVD,axis = 1)]).transpose()
plots.plot_method_error(error_ratios.transpose(), np.arange(iterations), xlabel = "Greedy Sampling Iterations", ylabel = "Error Ratio to SVD", legend= ("Randomized","Single View"),
                        save_path = "Ma798_SingleViewROM/figures/GreedySampling/convergence_ratio.pdf",figsize = (5,3.5))

plots.plot_method_error(error_SSQ.transpose(), np.arange(iterations), xlabel = "Greedy Sampling Iterations", ylabel = "Error", legend= ("Determinisitc","Randomized","Single View"),
                        save_path = "Ma798_SingleViewROM/figures/GreedySampling/convergence_error.pdf",figsize = (5,3.5))


