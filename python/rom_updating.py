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
from fluid_rom import *

def IterativeOptimization(snapshots_used, snapshots_full, times, dx, n_added=5, iterations=5, method="SVD", modes=2):
    for i in range(iterations):
        #Form base ROM from intit
        if method == "SVD":
            (spatial,)  = POD(snapshots_used,modes)
        if method == "SingleView":
            (spatial, ) = POD(snapshots_used,modes)
        #Compute temporal modes (need to do this since spatial snapshots aren't neccesarily in sequence)
        temporal = np.matmul(spatial.transpose(), snapshots_full)
        #Form ROM
        rom_matrices = MakeFluidMatrices(spatial,dx,verbosity=0)
        #Compute ROM error
        error = ComputeROMerror(rom_matrices,temporal, snapshots_full, times)
        #Add snapshots
        snapshots_used = UpdateSnapshots(snapshots_used, snapshots_full, error, n_added)
        
def ComputeROMerror(rom_matrices,temporal,snapshots_full,times):
    # Compute multiplier
    
    # Add multiplier to matrices
    
    #Compute ROM solution across t
    
    #Compute error (as n_snapshots vector)
    
    return error

def UpdateSnapshots(snapshots_used, snapshots_full, error,n_added):
    # Get n_added tsteps with highest absolute error
    
    # Append new snapshots to snapshots_used
    return snapshots_used