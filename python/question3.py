#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Homework1/Question3.py
@Time    :   2023/09/18 13:57:17
@Author  :   Harley Hanes 
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2023, Harley Hanes
@Desc    :   None
'''

#import sys
from POD import *
import scipy 
import matplotlib.pyplot as plt
import numpy as np
import time
#from Functions.SVD import deterministic
#Supporting Functions, move to SVD module when issue figured out




verbosity = 1
mode_multiplier = 3
#Load Data
mat_file = scipy.io.loadmat("ma798fall2023/notebooks/mat/olivettifaces.mat")
faces=mat_file["faces"]
(m,n)=faces.shape
print("Matrix Dimension: " + str(faces.shape))
#Set hyper parameters
k_vec=np.arange(25,201,25)
p= 10
q=1
#Initialize_data_vectors
methods = (lambda matrix: POD(matrix, k, verbosity = verbosity),
            lambda matrix: randSVD(matrix,k,p,q),
            lambda matrix: SingleViewWrapper(matrix,k,1,mode_multiplier,verbosity = verbosity),
            lambda matrix: SingleViewWrapper(matrix,k,4,mode_multiplier,verbosity = verbosity))
         
         
    
error=np.empty((k_vec.size, len(methods)))
run_times = np.empty((k_vec.size,len(methods)-1))

for ik in np.arange(0,k_vec.size,1,dtype=int):
    k=k_vec[ik]
    l1=2*k+1
    l2=2*l1+1
    A_approx=np.empty((m,n,len(methods)))
    duration = np.empty((len(methods),))
    if k ==100:
        singular_values=np.empty((4,100))
    for i in np.arange(0,len(methods)):
        spatial, temporal = methods[i](faces)
        if verbosity ==2:
            print("spatial.shape: " + str(spatial.shape))
            print("temporal.shape: " + str(temporal.shape))
        A_approx[:,:,i]=spatial @ temporal.transpose()
        if k==100:
            singular_values[i,:]=np.sqrt(np.sum(temporal**2,axis=0))
    #Compute Errors
    for i in np.arange(0,len(methods)):
        if verbosity ==2:
            print("A Low-Rank head: "+ str(A_approx[0:5,0:5,i+1]))
        error[ik,i] = np.linalg.norm(faces-A_approx[:,:,i], ord=2)/np.linalg.norm(faces,ord=2)
    if (verbosity ==2 or verbosity=='debug'):
        print("Error: " + str(error))

#============================================== Plot Data =========================================
#Error
methods = ['Randomized', 'Single-View']
fig1=plt.figure()
plt.semilogy(k_vec, error)
plt.xlabel(r"k")
plt.ylabel("Relative Error of A")
#plt.ylabel(r"Relative Error: $\frac{\|A_{\text{low rank}}-A_{\text{method}}\|_2}{\|A_{\text{low rank}}\|_2}")
plt.legend(['Deterministic','Randomized', 'Single-View (1 interation)', 'Single-View (4 iterations)'])
plt.tight_layout()
plt.show()
fig1.savefig("Ma798_SingleViewROM/Figures/POD/Q3error.pdf")


fig3=plt.figure()
plt.semilogy(np.arange(1,101), singular_values.transpose())
plt.xlabel(r"i")
plt.ylabel(r"$\sigma_i$")
plt.legend(['Deterministic','Randomized', 'Single-View (1 interation)', 'Single-View (4 iterations)'])
plt.tight_layout()
plt.show()
fig3.savefig("Ma798_SingleViewROM/Figures/POD/Q3s.pdf")


        #print(RunTime)
#Solve Decompositions for each Method
#Outputs: RunTimes, U S (vec-form) V


#Compute Errors for each Decomposition

#Plot relative error (semilogy)

#Plot RunTimes

#Plot First 100 SV



