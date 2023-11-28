#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Aa798_SingelViewROA/python/rom_updating.py
@Time    :   2023/11/16 9:42:17
@Author  :   Harley Hanes 
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2023, Harley Hanes
@Desc    :   None
'''

import unittest
import numpy as np
import scipy 
import plots
from POD import *
from fluid_rom import *
import matplotlib.pyplot as plt

def IterativeOptimization(snapshots, snapshots_full, indices,modes, times, times_full, dx, batchsize=5, iterations=5, method="SVD",p=5,q=2):

    if method == "SVD": spatial  = POD(snapshots,modes)[0]
    if method == "RandSVD": spatial  = randPOD(snapshots,modes)[0]
    if method == "SingleView": spatial,storage = singleview(snapshots,modes)[0,2]
    
    for i in range(iterations):
        print("iteration", i+1)

        #Compute temporal modes (need to do this since spatial snapshots aren't neccesarily in sequence)
        
        temporal = (np.dot(spatial.T, snapshots_full)).T
        
        #Form ROM

        a0 = temporal[0]

        a_rom = ComputeROM(spatial,temporal,dx,times_full,a0)

        # Compute ROM error and update time snapshots
        
        error = ComputeROMerror(spatial,a_rom,snapshots_full)

        new_snapshots,new_times,new_indices = UpdateSnapshots(snapshots_full,times_full,error, batchsize)
        times = np.concatenate([times, new_times])
        indices = np.concatenate([indices, new_indices])

        if method == "SVD":
            snapshots = np.concatenate([snapshots,new_snapshots],axis=1)
            spatial  = POD(snapshots,modes)[0]
        if method == "RandSVD":
            snapshots = np.concatenate([snapshots,new_snapshots],axis=1)
            spatial  = randPOD(snapshots,modes)[0]
        if method == "SingleView":
            spatial,storage = update_singleview(new_snapshots,storage)[0,2]
    print("Selected Indices", sorted(indices))

def ComputeROM(spatial,temporal,dx,times,a0):
    rom_matrices = MakeFluidMatrices(spatial,dx,verbosity=0)
    scaling = ComputeDydtScaling(spatial, temporal, rom_matrices,times)
    t,a_rom,solver_output= SolveROM(rom_matrices,times,a0, dydt_scaling = scaling, verbosity=0) 
    return a_rom

        
def ComputeROMerror(spatial,a_rom,snapshots_full):
    error_matrix = snapshots_full - np.dot(spatial,a_rom)
    xs = np.linspace(0,10,1601)

    plt.plot(xs,error_matrix[:,0])
    plt.show()
    
    error = np.sum(error_matrix**2,axis=0)    
    return error

def UpdateSnapshots(snapshots_full, times_full,error,batchsize):
    print("TOP ERRORS", sorted(error)[-batchsize:])
    indices = sorted(range(len(error)), key=lambda x: error[x])[-batchsize:]
    new_snapshots = snapshots_full[:,indices]
    new_times = times_full[indices]
    return new_snapshots,new_times,indices

class CheckUpdate(unittest.TestCase):
    def test_updatesnapshot(self):
        v = np.array([1.2, 2.3, 3.2, 1.9, 4.0])
        A = np.identity(5)
        A1 = A[:,3,5]
        result = updateSnapshots(A,v,2)
        self.assertEqual(result, A1)
