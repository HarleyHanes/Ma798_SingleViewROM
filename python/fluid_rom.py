#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Ma798_SingelViewROM/python/fluid_rm.py
@Time    :   2023/11/16 9:42:17
@Author  :   Harley Hanes 
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2023, Harley Hanes
@Desc    :   None
'''

import unittest
import numpy as np

def H2(Phi, W=1.0, verbosity=0):
    """
    Computes the matrices of inner products (Phi_i Phi_j,Phi_k) for a given Phi matrix.

    Args:
        Phi: A numpy array of size n by k, where n is the number of data points and k is the number of modes.
        W: Optional. The weight parameter to be passed to WeightedNorm. Default is 1.0.
        verbosity: Optional. The verbosity level to be passed to WeightedNorm. Default is 0.

    Returns:
        A numpy array of size k by k by k, where the ijk element is the output of WeightedNorm(Phi[:,i]*Phi[:,j],Phi[:,k],W).
    """

    if verbosity > 0:
        #Print size of Phi
        print("Phi.shape: ", Phi.shape)
    if verbosity > 1:
        #Print head of Phi
        print("Phi[:,:5]: ", Phi[:,:5])
        
    # Check that Phi is a numpy array of size n by k.
    CheckNumpy(Phi)

    # Compute the h2 matrix.
    mat = np.zeros((Phi.shape[1], Phi.shape[1], Phi.shape[1]))
    for i in range(Phi.shape[1]):
        for j in range(Phi.shape[1]):
            for k in range(Phi.shape[1]):
                mat[i, j, k] = WeightedNorm(Phi[:, i] * Phi[:, j], Phi[:, k], W=W, verbosity=verbosity)

    return mat


        
class TestH2(unittest.TestCase):
    def test_H2(self):
        # Generate a random Phi matrix
        np.random.seed(0)
        n = 10
        k = 3
        Phi = np.random.rand(n, k)

        # Compute the H2 matrix
        mat = H2(Phi)

        # Check the dimensions of the H2 matrix
        self.assertEqual(mat.shape, (k, k, k))

        # Check that the H2 matrix is symmetric
        for i in range(k):
            for j in range(k):
                for l in range(k):
                    self.assertAlmostEqual(mat[i, j, l], mat[j, i, l])
                    
                    
    def test_H2_with_Phi_2_1_1_0(self):
        # Create a Phi matrix
        Phi = np.array([[2, 1], [1, 0]])

        # Compute the H2 matrix
        mat = H2(Phi)

        # Check the dimensions of the H2 matrix
        self.assertEqual(mat.shape, (2, 2, 2))

        # Check that the H2 matrix is symmetric
        for i in range(2):
            for j in range(2):
                for l in range(2):
                    self.assertAlmostEqual(mat[i, j, l], mat[j, i, l])

        # Check that mat is equal to [[[9, 4], [4, 2]], [[4, 2], [2, 1]]]
        self.assertEqual(np.allclose(mat, [[[9, 4], [4, 2]], [[4, 2], [2, 1]]]), True)
                    
    





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
        verbosity = 1
        result = WeightedNorm(v1, v2, W, verbosity)
        self.assertEqual(result, 32)

    def test_invalid_inputs(self):
        v1 = np.array([1, 2, 3])
        v2 = [4, 5, 6]
        W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        verbosity = 1
        self.assertRaises(TypeError, WeightedNorm, v1, v2, W, verbosity)

        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        W = np.array([[1, 0, 0], [0, 1, 0]])
        verbosity = 1
        self.assertRaises(ValueError, WeightedNorm, v1, v2, W, verbosity)

    def test_valid_inputs_w_1(self):
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        W = 1.0
        verbosity = 1
        result = WeightedNorm(v1, v2, W, verbosity)
        self.assertEqual(result, 32)

if __name__ == '__main__':
    unittest.main()