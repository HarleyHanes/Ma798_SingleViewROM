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

#Set n snapshots and n modes

#Load data
location = "Ma798_SingleViewROM/Fiber_Data/"
(phi,a,times,xs)= data.load(location, time_steps=np.arange(60,202), verbosity=1)
a0 = a[0,:]
#Form ROM
matrices = fluid_rom.MakeFluidMatrices(phi,xs[1]-xs[0], verbosity=2)
fluid_rom.CheckDydt(a,phi, matrices,times, folder = "Ma798_SingleViewROM/figures/CheckDydt/", verbosity=2, plot = True)
#Solve ROM
scaling = fluid_rom.ComputeDydtScaling(phi, a, matrices,times)
print("scaling: ", scaling)
(t,a_rom,solver_output)=fluid_rom.SolveROM(matrices,times,a0, dydt_scaling = scaling, verbosity=1)
a_rom=a_rom.transpose()
a=a[0:a_rom.shape[0],:]
#Compute ROM Error
total_error = np.linalg.norm(a-a_rom)/np.linalg.norm(a)
time_error = np.sum(((a-a_rom)/a)**2, axis =1)
#print("Error: ", total_error)
#print("a-a_rom: ", a-a_rom)
#print("time_error: ", time_error)
#Plot ROM Solution
#plots.plot_solution(np.matmul(phi,a_rom),xs,t)

plots.plot_temporal(a_rom,t,a_rom.shape[1],xlabel="t",ylabel="a", title="a(t) Computed with Projection ROM", 
                    save_path= "Ma798_SingleViewROM/figures/ROMoutputs/a_rom")
plots.plot_temporal(a,t,a_rom.shape[1],xlabel="t",ylabel="a", title="True a(t)", 
                    save_path= "Ma798_SingleViewROM/figures/ROMoutputs/a_true")

plots.plot_error(a[1:,:],a_rom[1:,],t[1:],logy=True, title="Relative ROM Error",xlabel='t', ylabel='Error', 
                    save_path= "Ma798_SingleViewROM/figures/ROMoutputs/ROM_error")

