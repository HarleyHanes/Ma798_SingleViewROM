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

def IterativeOptimization(snapshots, snapshots_full, indices,modes, times, times_full, dx, batchsize=5, iterations=5, method="SVD",p=5,q=2, verbosity =0):

    if method == "SVD": spatial  = POD(snapshots,modes)[0]
    if method == "RandSVD": spatial  = randSVD(snapshots,modes,p,q)[0]
    if method == "SingleView": spatial,trash, storage = singleview(snapshots,modes)
    error=np.empty((iterations,snapshots_full.shape[1]))
    for i in range(iterations):
        if verbosity > 0:
            print("iteration", i+1)

        #Compute temporal modes (need to do this since spatial snapshots aren't neccesarily in sequence)
        
        temporal = (np.dot(spatial.T, snapshots_full)).T
        
        #Form ROM

        a0 = temporal[0]

        a_rom = ComputeROM(spatial[:,1:],temporal[:,1:],dx,times_full,a0[1:])

        first_temporal = np.reshape(temporal[:,0], (temporal.shape[0],1))
        #Append first mode to a_rom
        a_rom_combined = np.append(first_temporal,a_rom, axis=1)
        # Compute ROM error and update time snapshots
        
        error[i,:] = ComputeROMerror(spatial,a_rom_combined,snapshots_full,temporal,verbosity = verbosity)
        indices_unused = np.delete(np.arange(len(error[i,:]), dtype=int),indices)
        new_snapshots,new_times,new_indices = UpdateSnapshots(snapshots_full,times_full,np.delete(error[i,:],indices),
                                                              indices_unused,batchsize,verbosity = verbosity)
        
        if verbosity > 1:
            print("indices.shape: ", indices.shape)
            print("new_indices.shape: ", new_indices.shape)
        times = np.concatenate([times, new_times])
        indices = np.concatenate([indices, new_indices])

        snapshots = np.concatenate([snapshots,new_snapshots],axis=1)
        if method == "SVD":
            spatial  = POD(snapshots,modes)[0]
        if method == "RandSVD":
            spatial  = randSVD(snapshots,modes,p,q)[0]
        if method == "SingleView":
            spatial,trash,storage = update_singleview(new_snapshots,storage)
    print("Selected Indices", sorted(indices))
    return (error, indices)

def ComputeROM(spatial,temporal,dx,times,a0):
    rom_matrices = MakeFluidMatrices(spatial,dx,verbosity=0)
    #scaling = ComputeDydtScaling(spatial, temporal, rom_matrices,times)
    t,a_rom,solver_output= SolveROM(rom_matrices,times,a0, verbosity=0) 
    return a_rom.transpose()

        
def ComputeROMerror(spatial,a_rom,snapshots_full,a_true,verbosity = 0, relative = True, error_type = "snapshot"):
    if error_type == "POD":
        error_matrix = a_rom-a_true
    else :
        error_matrix = snapshots_full - np.dot(spatial,a_rom.transpose())
    if verbosity>0:
        print("error_matrix.shape: ", error_matrix.shape)
        print("np.linalg.norm(error_matrix): ", np.linalg.norm(error_matrix))
    if verbosity >1:
       plt.plot(error_matrix)
       plt.show()
    
    if relative:
        if error_type =="POD":
            error = np.sqrt(np.sum(error_matrix**2,axis=1)/np.sum(a_true**2,axis =1 ))
        else :
            error = np.sqrt(np.sum(error_matrix**2,axis=0)/np.sum(snapshots_full**2,axis=0))    
    else :
        if error_type =="POD":
            error = np.sqrt(np.sum(error_matrix**2,axis=1))
        else :
            error = np.sqrt(np.sum(error_matrix**2,axis=0))
    return error

def UpdateSnapshots(snapshots_full, times_full,error, indices_unused,batchsize, verbosity =0):

    snapshots_used = len(times_full)-len(indices_unused)
    if verbosity > 0:
        print("TOP ERRORS", sorted(error)[-batchsize:])
    sorted_indices = sorted(indices_unused, key = lambda x: error[x-snapshots_used])
    indices = sorted_indices[-batchsize:]
    new_snapshots = snapshots_full[:,indices]
    new_times = times_full[indices]
    return new_snapshots,new_times,np.array(indices)

class CheckUpdate(unittest.TestCase):
    def test_updatesnapshot(self):
        v = np.array([1.2, 2.3, 3.2, 1.9, 4.0])
        A = np.identity(5)
        A1 = A[:,3,5]
        result = UpdateSnapshots(A,v,2)
        self.assertEqual(result, A1)
