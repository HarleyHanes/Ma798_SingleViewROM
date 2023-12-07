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
    #if energy != "null":
    print("Energy Quantified by k =", modes, ": ", np.sum(S[:modes]/np.sum(S)))
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
    

def singleview(A, modes,verbosity=0, mode_multiplier = 6):

    if verbosity > 0:
        print("A.shape: ", A.shape)
        print("modes: ", modes)
    if verbosity > 1:
         print("A[:,:5]: ", A[:,:5])
         print("modes: ", modes)  

    l1=mode_multiplier*modes+1
    l2=mode_multiplier*l1+1
    m,n = A.shape
    
    Omega = np.random.randn(n, l1)
    Psi   = np.random.randn(l2, m)

    Y = np.dot(A,Omega)
    W = np.dot(Psi,A)
    
    Q = np.linalg.qr(Y, mode = 'reduced')[0]

    X = np.dot(Psi,Q)

    B = np.linalg.lstsq(X,W,rcond=None)[0]
    B = np.dot(Q,B)

    spatial,temporal = POD(B,modes)

    storage = {"Psi": Psi, "Y": Y, "W": W,"modes":modes, "l1": l1, "l2": l2}
    
    return spatial,temporal,storage

def update_singleview(A,storage,verbosity=0):

    if verbosity > 0:
        print("A.shape: ", A.shape)
        print("Stored variables: ",list(storage))
    if verbosity > 1:
         print("A[:,:5]: ", A[:,:5])
         
    m,n = A.shape
    
    Omega = np.random.randn(n, storage["l1"])

    Y = storage["Y"] + np.dot(A,Omega)
    W = np.concatenate([storage["W"],np.dot(storage["Psi"],A)],axis=1)

    Q = np.linalg.qr(Y, mode = 'reduced')[0]
    
    X = np.dot(storage["Psi"],Q)
                       
    B = np.linalg.lstsq(X,W,rcond=None)[0]
    B = np.dot(Q,B)

    spatial,temporal = POD(B,storage["modes"])

    storage.update({"Y": Y, "W": W})

    return spatial,temporal,storage
    

def SingleViewWrapper(matrix,k,n_updates,mode_multiplier,verbosity = 0):
    #split columns of matrix into n_updates
    matrix_split = np.split(matrix,n_updates,axis=1)
    if verbosity >0:
        print("len(matrix_split)", len(matrix_split))
    #make initialization matrix
    first_matrix=np.zeros(matrix.shape)
    if verbosity >0:
        print("matrix.shape[1]/n_updates: ", matrix.shape[1]/n_updates)
    first_matrix[:,:int(matrix.shape[1]/n_updates)]=matrix_split[0]
    #run first singleView
    spatial,temporal,storage = singleview(matrix_split[0], k, mode_multiplier= mode_multiplier, verbosity=0)
    if verbosity>0:
        print("spatial.shape: ", spatial.shape)
        print("temporal.shape: ", temporal.shape)
    #run updates
    for i in range(n_updates-1):
        spatial,temporal, storage = update_singleview(matrix_split[i+1],storage)
        if verbosity>0:
            print("matrix_split[i+1].shape: ", matrix_split[i+1].shape)
            print("spatial.shape: ", spatial.shape)
            print("temporal.shape: ", temporal.shape)
    return spatial,temporal