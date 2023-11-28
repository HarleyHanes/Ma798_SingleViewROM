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
        
    
def plot_singular_values(S):

    plt.plot(S)
    plt.xlabel("k")
    plt.ylabel("s_k")
    plt.show()

def plot_temporal(temporal,times,k, xlabel = "null", ylabel = "null", title = "null", save_path = "null"):

    ks = int(np.ceil(k**.5))
    ks2 = ks

    if (ks*(ks2-1) >= k):
        ks2 = ks2-1
    fig, axs= plt.subplots(ks,ks2)
    print("ks", ks)
    print("ks2", ks2)
    if k>3:
        for i in range(k):
            ax_iterate=[i // ks2 , i % ks2]
            axs[ax_iterate[0], ax_iterate[1]].plot(times,temporal[:,i])
            if xlabel != "null" and (i>=(ks-1)*ks2): 
                axs[ax_iterate[0], ax_iterate[1]].set_xlabel(xlabel)
            if ylabel != "null" and (i % ks2 ==0):
                axs[ax_iterate[0], ax_iterate[1]].set_ylabel(ylabel)
        #Title feature not working now. Need to figure out how to put title over whole fig rather than just in subplot
    elif k>1 : 
        for i in range(k):
            axs[i].plot(times,temporal[:,i])
            if xlabel != "null" and (i==0): 
                axs[i].set_xlabel(xlabel)
            if ylabel != "null":
                axs[i].set_ylabel(ylabel)
        #Title feature not working now. Need to figure out how to put title over whole fig rather than just in subplot
    else :
        axs.plot(times,temporal)
        if xlabel != "null": 
            axs.set_xlabel(xlabel)
        if ylabel != "null":
            axs.set_ylabel(ylabel)
        #Title feature not working now. Need to figure out how to put title over whole fig rather than just in subplot
    if title != "null":
        plt.suptitle(title)
    if save_path == "null":
        plt.show()
    else :
        plt.savefig(save_path)
        plt.show()
        
def plot_error(temporal_true, temporal_rom, times, logy= False,
               xlabel = "null", ylabel = "null", title = "null",  save_path = "null"):
        
    #time_error = np.sum(np.sqrt(((temporal_true-temporal_rom)/temporal_true)**2), axis =1)

    time_error = np.sqrt(np.sum((temporal_true-temporal_rom)**2, axis =1))/np.sqrt(np.sum(temporal_true**2, axis =1))
    plt.plot(times,time_error)
    if logy:
        plt.yscale("log")
    if xlabel != "null":
        plt.xlabel(xlabel)
    if ylabel != "null":
        plt.ylabel(ylabel)
    if title != "null":
        plt.title(title)
    
    if save_path == "null":
        plt.show()
    else :
        plt.savefig(save_path)
        plt.show()
