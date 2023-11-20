#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Aa798_SingelViewROA/python/fluid_rm.py
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



    
def MakeFluidMatrices(Phi):
    """
    Incomplete
    Args:
    
    
    Returns:
    """
    #Compute derivatives
    dPhidx = ...
    d2Phidx = ...
    d3Phidx= ...
    d4Phidx=...
    
    #Make matrices
    h2_dhdx=InnerProd3rdOrder(Phi, Phi,Phi,dPhidx)
    h2_dhdx2=InnerProd4thOrder(Phi, Phi,Phi,dPhidx,dPhidx)
    h2_dhdx_d3hdx=InnerProd4thOrder(Phi, Phi,Phi,dPhidx,d3Phidx)
    
    h3_d2hdx = InnerProd4thOrder(Phi, Phi,Phi,Phi,d2Phidx)
    h3_d4hdx = InnerProd4thOrder(Phi, Phi,Phi,Phi,d4Phidx)
    
    return {"h2_dhdx": h2_dhdx,"h2_dhdx2": h2_dhdx2, "h2_dhdx_d3hdx": h2_dhdx_d3hdx, "h3_d2hdx": h3_d2hdx, "h3_d4hdx": h3_d4hdx}
    
    
def ROMdydt (t,a,ThirdOrders,FourthOrders):
    
    # Implementation of Formula dydt_i=-aj*ak*al*[ThirdOrders_ijkl+am*(FourthOrders_ijklm)] (using einstein's notation)
    dydt=np.dot(FourthOrders,a)
    dydt = np.dot(ThirdOrders+dydt,a)
    dydt= np.dot(dydt,a)
    dydt = np.dot(dydt,a)
    
    return dydt

def SolveROM(matrices,a0,t_input, method='RK45',verbosity =0):
    h2_dhdx= matrices["h2_dhdx"]
    h2_dhdx2= matrices["h2_dhdx2"]
    h2_dhdx_d3hdx= matrices["h2_dhdx_d3hdx"]
    h3_d2hdx= matrices["h3_d2hdx"]
    h3_d4hdx= matrices["h3_d4hdx"]
    ThirdOrders=-(3/2*h2_dhdx)
    FourthOrders=-(3*h2_dhdx2+3*h2_dhdx_d3hdx+h3_d2hdx+h3_d4hdx)
    if verbosity > 0:
        print("h2_dhdx.size: ", h2_dhdx.size)
        print("FourthOrders.size: ", h2_dhdx.size)
        
    #Check t_span is 1D array
    
    if t_input.size<2 or t_input.ndim!=1:
        raise Exception("Need a 1D array of at least size 2 for t_input")
    elif t_input.size==2:
        t_span=t_input
        scipy_outputs = scipy.integrate.solve_ivp(ROMdydt,
                                                a0,
                                                t_span,
                                                args = (ThirdOrders, FourthOrders),
                                                method = method)
    else:
        t_eval=t_input
        t_span =[t_input[0], t_input[-1]]
        scipy_outputs = scipy.integrate.solve_ivp(ROMdydt,
                                                a0,
                                                t_span,
                                                args = (ThirdOrders, FourthOrders),
                                                method = method,
                                                t_eval=t_eval)
    
    return(scipy_outputs.t, scipy_outputs.y, scipy_outputs)
    
    
   
def InnerProd4thOrder(Ar, Al1,Al2,Al3,Al4,  W=1.0, verbosity =0):
    """
    Computes the matrices of inner products (Ar_m, Al1_i Al2_j Al3_k Al4_l).

    Args:
        Ar: RHS inner product matrix,  numpy array of size n by k.
        Al1: First LHS inner product matrix,  numpy array of size n by k.
        Al2: Second LHS inner product matrix,  numpy array of size n by k.
        Al3: Third LHS inner product matrix,  numpy array of size n by k.
        Al4: Fourth LHS inner product matrix,  numpy array of size n by k.
        W: Optional. The weight parameter to be passed to WeightedNorm. Default is 1.0.
        verbosity: Optional. The verbosity level to be passed to WeightedNorm. Default is 0.

    Returns:
        A numpy array of size k by k by k by k by k, where the ijklm element is the output of WeightedNorm(Al1[:,j]*Al2[:,k]*Al3[:,l]*Al4[:,m], Ar[:,i],W).
    """

    if verbosity > 0:
        #Print size of A
        print("Ar.shape: ", Ar.shape)
        print("Al1.shape: ", Al1.shape)
        print("Al2.shape: ", Al2.shape)
        print("Al3.shape: ", Al3.shape)
        print("Al4.shape: ", Al4.shape)
    if verbosity > 1:
        #Print head of Phi
        print("Ar[:,:5]: ", Ar[:,:5])
        print("Al1[:,:5]: ", Al1[:,:5])
        print("Al2[:,:5]: ", Al2[:,:5])
        print("Al3[:,:5]: ", Al3[:,:5])
        print("Al4[:,:5]: ", Al4[:,:5])
        
    # Check that Phi is a numpy array of size n by k.
    CheckNumpy(Ar)
    CheckNumpy(Al1,dim=Ar.shape)
    CheckNumpy(Al2,dim=Ar.shape)
    CheckNumpy(Al3,dim=Ar.shape)
    CheckNumpy(Al4,dim=Ar.shape)

    # Compute the h2 matrix.
    mat = np.zeros((Al1.shape[1], Al2.shape[1], Al3.shape[1],Al4.shape[1], Ar.shape[1]))
    for i in range(Ar.shape[1]):
        for j in range(Al1.shape[1]):
            for k in range(Al2.shape[1]):
                for l in range(Al3.shape[1]):
                    for m in range(Al4.shape[1]):
                        mat[i, j, k, l, m] = WeightedNorm(Al1[:, j] * Al2[:, k]* Al3[:, l]* Al4[:, m], Ar[:, i], W=W, verbosity=verbosity)

    return mat

def InnerProd3rdOrder(Ar, Al1,Al2,Al3, W=1.0, verbosity =0):
    """
    Computes the matrices of inner products (Ar_l, Al1_i Al2_j Al3_k).

    Args:
        Ar: RHS inner product matrix,  numpy array of size n by k.
        Al1: First LHS inner product matrix,  numpy array of size n by k.
        Al2: Second LHS inner product matrix,  numpy array of size n by k.
        Al3: Third LHS inner product matrix,  numpy array of size n by k.
        W: Optional. The weight parameter to be passed to WeightedNorm. Default is 1.0.
        verbosity: Optional. The verbosity level to be passed to WeightedNorm. Default is 0.

    Returns:
        A numpy array of size k by k by k by k, where the ijkl element is the output of WeightedNorm(Al1[:,j]*Al2[:,k]*Al3[:,l], Ar[:,i],W).
    """

    if verbosity > 0:
        #Print size of A
        print("Ar.shape: ", Ar.shape)
        print("Al1.shape: ", Al1.shape)
        print("Al2.shape: ", Al2.shape)
        print("Al3.shape: ", Al3.shape)
    if verbosity > 1:
        #Print head of Phi
        print("Ar[:,:5]: ", Ar[:,:5])
        print("Al1[:,:5]: ", Al1[:,:5])
        print("Al2[:,:5]: ", Al2[:,:5])
        print("Al3[:,:5]: ", Al3[:,:5])
        
    # Check that Phi is a numpy array of size n by k.
    CheckNumpy(Ar)
    CheckNumpy(Al1,dim=Ar.shape)
    CheckNumpy(Al2,dim=Ar.shape)
    CheckNumpy(Al3,dim=Ar.shape)

    # Compute the h2 matrix.
    mat = np.zeros((Ar.shape[1], Al1.shape[1], Al2.shape[1], Al3.shape[1]))
    for i in range(Ar.shape[1]):
        for j in range(Al1.shape[1]):
            for k in range(Al2.shape[1]):
                for l in range(Al3.shape[1]):
                    mat[i, j, k, l] = WeightedNorm(Al1[:, j] * Al2[:, k]* Al3[:, l], Ar[:, i], W=W, verbosity=verbosity)

    return mat
                    
class TestInnerProd3rdOrder(unittest.TestCase):

    def test_InnerProd3rdOrder_with_Phi_2_1_1_0(self):
        # Create a Phi matrix
        Phi = np.array([[2, 1], [1, 0]])
        I = np.array([[1, 0], [0, 1]])
        # Compute the H2 matrix
        mat = InnerProd3rdOrder(Phi, Phi,Phi,I)

        # Check the dimensions of the H2 matrix
        self.assertEqual(mat.shape, (2, 2, 2, 2))

        # Check that mat is equal to [[[9, 4], [4, 2]], [[4, 2], [2, 1]]]
        a=np.array([[[[8,4],[1,0]],
                     [[4,2],[0,0]]],
                    [[[4,2],[0,0]],
                     [[2,1],[0,0]]]])
        
        
        np.testing.assert_array_equal(mat,np.transpose(a,axes = [3,0,1,2]))




def WeightedNorm(v1,v2, W=1, verbosity = 0):
    #Check Inputs v1, v2 are numpy arrays, print error message if they are not
    CheckNumpy(v1)
    CheckNumpy(v2,v1.shape)
    
    if verbosity > 0:
        #Print sizes of v1 and v2
        print("v1.shape: ", v1.shape, "v2.shape: ", v2.shape)
    if verbosity > 1:
        #Print first 5 elements of v1 and v2
        print("v1[:5]: ", v1[:5], "v2[:5]: ", v2[:5])
        
    #Check W
    if not isinstance(W, np.ndarray):
        if not isinstance(W, float):
            raise TypeError("W must be a float or a numpy array")
    elif (W.shape[0] != v1.size) & (W.shape[1]!= v2.size) :
        raise ValueError("W must have dimensions equal to v1 X v2")
                         
                         
    norm = np.dot(np.dot(v1, W), v2)
    return norm


def CheckNumpy(v, dim=None):
    if not isinstance(v, np.ndarray):
        raise TypeError("v must be a numpy array")
    if dim is not None:
        if v.shape != dim:
            raise ValueError("v must have shape {}".format(dim))




class TestCheckNumpy(unittest.TestCase):
    def test_valid_numpy_array(self):
        v = np.array([1, 2, 3])
        self.assertIsNone(CheckNumpy(v))

    def test_invalid_numpy_array(self):
        v = [1, 2, 3]
        self.assertRaises(TypeError, CheckNumpy, v)

    def test_valid_numpy_array_with_shape(self):
        v = np.array([[1, 2], [3, 4]])
        dim = (2, 2)
        self.assertIsNone(CheckNumpy(v, dim))

    def test_invalid_numpy_array_with_shape(self):
        v = np.array([[1, 2], [3, 4]])
        dim = (2, 3)
        self.assertRaises(ValueError, CheckNumpy, v, dim)
        

class TestWeightedNorm(unittest.TestCase):
    def test_valid_inputs(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = WeightedNorm(v1, v2, W)
        self.assertEqual(result, 32)

    def test_invalid_inputs(self):
        v1 = np.array([1, 2, 3])
        v2 = [4, 5, 6]
        W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        verbosity = 1
        self.assertRaises(TypeError, WeightedNorm, v1, v2, W)

        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        W = np.array([[1, 0, 0], [0, 1, 0]])
        verbosity = 1
        self.assertRaises(ValueError, WeightedNorm, v1, v2, W)

    def test_valid_inputs_w_1(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        W = 1.0
        verbosity = 1
        result = WeightedNorm(v1, v2, W, verbosity)
        self.assertEqual(result, 32)

if __name__ == '__main__':
    unittest.main()