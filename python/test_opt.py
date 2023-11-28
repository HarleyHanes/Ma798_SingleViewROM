import numpy as np
import fluid_rom
import plots
import data
from rom_updating import *

#Set n snapshots and n modes

method = "SVD"
#Load data

location = ""
ts = 60
te = 201

b0 = 10
batchsize = 10
Niters = 10
p=5
q=2

time_steps = np.arange(ts,te)

(data_full,spatial,temporal,aves,times,xs)= data.load(location, time_steps, verbosity=0)

t0s = np.linspace(0,te-ts-1,b0).astype(int)

data = data_full[:,t0s]

if method == "SingleView":
    padding = np.zeros(data.shape[0],Niters*batchsize)
    data = np.concatenate([data,padding],axis=1)
    
IterativeOptimization(data, data_full, t0s,2, times[t0s], times, xs[2]-xs[1], batchsize, Niters, method,p,q)
