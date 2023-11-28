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

def IterativeOptimization(snapshots, snapshots_full, averages, modes, times, dx, batchsize=5, iterations=5, method="SVD",p=5,q=2):

    if method == "SVD": spatial  = POD(snapshots,modes)[0]
    if method == "RandSVD": spatial  = randPOD(snapshots,modes)[0]
    if method == "SingleView": spatial,storage = singleview(snapshots,modes)[0,2]
    
    for i in range(iterations):
            
        #Compute temporal modes (need to do this since spatial snapshots aren't neccesarily in sequence)
        
        temporal = np.dot(spatial.T, snapshots_full)
        
        #Form ROM
        
        rom_matrices = MakeFluidMatrices(spatial,dx,verbosity=0)
        
        #Compute ROM error
        
        error = ComputeROMerror(rom_matrices,temporal, snapshots_full, times)

        new_snapshots = UpdateSnapshots(snapshots_full, error, batchsize)

        if method == "SVD":
            snapshots = np.concatenate([snapshots,new_snapshots])
            spatial  = POD(snapshots,modes)[0]
        if method == "RandSVD":
            snapshots = np.concatenate([snapshots,new_snapshots])
            spatial  = randPOD(snapshots,modes)[0]
        if method == "SingleView":
            spatial,storage = update_singleview(new_snapshots,storage)[0,2]
        
def ComputeROMerror(rom_matrices,temporal,snapshots_full,times):
    # Compute multiplier
    
    # Add multiplier to matrices
    
    #Compute ROM solution across t
    
    #Compute error (as n_snapshots vector)

    # error_matrix = _____________ (snapshots_full - spatial*a_rom)
    
    error = np.sum(error_matrix**2,axis=1)    
    return error

def UpdateSnapshots(snapshots_full, error,batchsize):
    indices = sorted(range(len(error)), key=lambda x: error[x])[-batchsize:]
    new_snapshots=snapshots_full[:,indices]
    return new_snapshots

class CheckUpdate(unittest.TestCase):
    def test_updatesnapshot(self):
        v = np.array([1.2, 2.3, 3.2, 1.9, 4.0])
        A = np.identity(5)
        A1 = A[:,3,5]
        result = updateSnapshots(A,v,2)
        self.assertEqual(result, A1)
