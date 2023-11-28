import numpy as np
import matplotlib.pyplot as plt

def POD(A, modes,verbosity=0):

    if verbosity > 0:
        print("A.shape: ", A.shape)
        print("modes: ", modes)
    if verbosity > 1:
         print("A[:,:5]: ", A[:,:5])
         print("modes: ", modes)  
        
    """ Computes the spatial modes and temporal coefficients using the POD """

    M,S,Vt =np.linalg.svd(A, full_matrices = False)

    spatial = M[:,:modes]
    temporal = np.dot(spatial.T,A)

    return spatial, temporal.T

def randSVD(A,k,p,q):

    m = A.shape[0]
    G = np.random.normal(0,1,size=(m,k+p))

    Q = np.linalg.qr(A.T @ G,mode='reduced')[0]

    for i in range(q):
        Q = np.linalg.qr(A @ Q,mode='reduced')[0]
        Q = np.linalg.qr(A.T @ Q,mode='reduced')[0]                     

    M,S,Vt = np.linalg.svd(A @ Q, full_matrices = False)
    V = Q @ (Vt[:k,:]).T

    spatial = M[:,:modes]
    temporal = np.dot(spatial.T,A)

    return spatial, temporal.T
    

def singleview(A, modes,verbosity=0):

    if verbosity > 0:
        print("A.shape: ", A.shape)
        print("modes: ", modes)
    if verbosity > 1:
         print("A[:,:5]: ", A[:,:5])
         print("modes: ", modes)  

    m,n = A.shape
    
    Omega = np.random.randn(n, 2*modes+1)
    Psi   = np.random.randn(2*(2*modes+1)+1, m)

    Y = np.dot(A,Omega)
    W = np.dot(Psi,A)
    
    Q = np.linalg.qr(Y, mode = 'reduced')[0]

    Q = Q[:,:modes]
    X = np.dot(Psi,Q)

    B = np.linalg.lstsq(X,W,rcond=None)[0]
    B = np.dot(Q,B)

    spatial,temporal = POD(B,modes)

    storage = {"Omega": Omega, "Psi": Psi, "Y": Y, "W": W,"Q": Q,"X":X,"modes":modes}

    return spatial,temporal,storage

def update_singleview(A,storage,verbosity=0):

    if verbosity > 0:
        print("A.shape: ", A.shape)
        print("Stored variables: ",list(storage))
    if verbosity > 1:
         print("A[:,:5]: ", A[:,:5])

    n = storage["Q"].shape[1]
    nend = n+storage["modes"]

    A2 = A[:,np.nonzero(np.any(A != 0, axis=0))[0]]
    
    Y = storage["Y"] + np.dot(A,storage["Omega"])
    W = storage["W"] + np.dot(storage["Psi"],A)

    Q = np.linalg.qr(np.concatenate([storage["Q"],A2],axis=1))[0]
    Q = Q[:,:nend]
    
    X = np.concatenate([storage["X"],np.dot(storage["Psi"],Q[:,n:])],axis=1)
    B = np.linalg.lstsq(X,W,rcond=None)[0]
    B = np.dot(Q,B)

    spatial,temporal = POD(B,storage["modes"])

    storage.update({"Y": Y, "W": W,"Q": Q,"X":X})

    return spatial,temporal,storage
    

