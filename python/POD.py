import numpy as np

def POD(A, modes,verbosity=0):

    if verbosity > 0:
        print("A.shape: ", A.shape)
        print("modes: ", modes)
    if verbosity > 1:
         print("A[:,:5]: ", A[:,:5])
         print("modes: ", modes)  
        
    """ Computes the spatial modes and temporal coefficients using the POD """
    averages = (np.sum(A,axis=0,keepdims=True)/len(A[:,0]))
    msA = A - averages
    
    M,S,Vt =np.linalg.svd(msA, full_matrices = False)

    spatial = M[:,:modes]
    temporal = np.dot(spatial.T,msA)

    return spatial, temporal.T
