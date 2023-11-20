import numpy as np
import matplotlib.pyplot as plt

def plot_solution(data,xs,times):
    for i in range(len(times)):
        plt.cla()
        plt.title('Time Snapshot at t = {}'.format(times[i]))
        plt.xlabel('x')
        plt.ylabel('h(x,t)')
        plt.ylim(np.min(np.min(data)), np.max(np.max(data)))
        plt.plot(xs,data[:,i])
        plt.pause(0.1)
    plt.show()
    
    
def plot_spatial(spatial,xs,k):
    
    ks = int(np.ceil(k**.5))
    ks2 = ks

    if (ks*(ks2-1) >= k):
        ks2 = ks2-1
    
    for i in range(k):
        plt.subplot(ks2,ks,i+1)
        plt.plot(xs,spatial[:,i])

def plot_temporal(temporal,times,k):

    ks = int(np.ceil(k**.5))
    ks2 = ks

    if (ks*(ks2-1) >= k):
        ks2 = ks2-1
    
    for i in range(k):
        plt.subplot(ks2,ks,i+1)
        plt.plot(times,temporal[:,i])
