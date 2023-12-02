import numpy as np

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

    spatial = M[:,:k]
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

    X = np.dot(Psi,Q)

    B = np.linalg.lstsq(X,W,rcond=None)[0]
    B = np.dot(Q,B)

    spatial,temporal = POD(B,modes)

    storage = {"Psi": Psi, "Y": Y, "W": W,"modes":modes}
    
    return spatial,temporal,storage

def update_singleview(A,storage,verbosity=0):

    if verbosity > 0:
        print("A.shape: ", A.shape)
        print("Stored variables: ",list(storage))
    if verbosity > 1:
         print("A[:,:5]: ", A[:,:5])
         
    m,n = A.shape

    Omega = np.random.randn(n, 2*storage["modes"]+1)

    Y = storage["Y"] + np.dot(A,Omega)
    W = np.concatenate([storage["W"],np.dot(storage["Psi"],A)],axis=1)

    Q = np.linalg.qr(Y, mode = 'reduced')[0]
    
    X = np.dot(storage["Psi"],Q)
                       
    B = np.linalg.lstsq(X,W,rcond=None)[0]
    B = np.dot(Q,B)

    spatial,temporal = POD(B,storage["modes"])

    storage.update({"Y": Y, "W": W})

    return spatial,temporal,storage
    

