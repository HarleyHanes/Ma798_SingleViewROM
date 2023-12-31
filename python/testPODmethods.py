import numpy as np
import fluid_rom
import plots
import data
import matplotlib.pyplot as plt
from POD import *

#Set n snapshots and n modes

#Load data
np.random.seed(0)
mode_multiplier=2
location = "Ma798_SingleViewROM/Fiber_Data/"
ts = 80
te = 200

b0 = 40
batchsize = 40
Niters = 2
p=5
q=1

modes = 2 # using 2 modes so the differences in methods are visable, everything is on top of each other with 4

time_steps = np.arange(ts,te)

(data_full,spatial,temporal,aves,times,xs)= data.load(location, time_steps, verbosity=0)

A1 = data_full[:,:40]
A2 = data_full[:,40:80]
A3 = data_full[:,80:]

# The first 40 columns

S1, T1  = POD(A1,modes)
randS1, randT1  = randSVD(A1,modes,p,q)
svS1, svT1, storage = singleview(A1,modes,mode_multiplier=mode_multiplier)

# Add 40 more columns

snapshots = np.concatenate([A1,A2],axis=1)

S2,T2  = POD(snapshots,modes)
randS2,randT2  = randSVD(snapshots,modes,p,q)
svS2,svT2,storage = update_singleview(A2,storage)

# Add the remaining columns

snapshots = np.concatenate([snapshots,A3],axis=1)

S3,T3  = POD(snapshots,modes)
randS3,randT3  = randSVD(snapshots,modes,p,q)
svS3,svT3,storage = update_singleview(A3,storage)

# Compute reconstructions

A = data_full + aves

B = np.dot(S1,T1.T)+aves[:40]
BR = np.dot(randS1,randT1.T)+aves[:40]
BS = np.dot(svS1,svT1.T)+aves[:40]

C = np.dot(S2,T2.T)+aves[:80]
CR = np.dot(randS2,randT2.T)+aves[:80]
CS = np.dot(svS2,svT2.T)+aves[:80]

D = np.dot(S3,T3.T)+aves
DR = np.dot(randS3,randT3.T)+aves
DS = np.dot(svS3,svT3.T)+aves

# Plot Reconstructions
width = 8
plt.figure(figsize=(width,width*.5))
plt.subplot(1,3,1)

plt.plot(xs,A[:,0],'k',label='True Solution')
plt.plot(xs,B[:,0],'r',label='POD Recon')
plt.plot(xs,BR[:,0],'g--',label='RandPOD Recon')
plt.plot(xs,BS[:,0],'b:',label='SingleView Recon')
plt.title("40 Snapshots")
plt.ylabel("h(t,x)")
plt.xlabel("x")

plt.subplot(1,3,2)

plt.plot(xs,A[:,0],'k',label='True Solution')
plt.plot(xs,C[:,0],'r',label='POD Recon')
plt.plot(xs,CR[:,0],'g--',label='RandPOD Recon')
plt.plot(xs,CS[:,0],'b:',label='SingleView Recon')
plt.title("80 Snapshots")

plt.subplot(1,3,3)

plt.plot(xs,A[:,0],'k',label='True Solution')
plt.plot(xs,D[:,0],'r',label='POD Recon')
plt.plot(xs,DR[:,0],'g--',label='RandPOD Recon')
plt.plot(xs,DS[:,0],'b:',label='SingleView Recon')
plt.title("120 Snapshots")
plt.suptitle("Reconstruction of $t="+ str(ts) + "$ with $(k,\ell_1, \ell_2)=("
             + str(storage["modes"])+ ", " +str(storage["l1"])+ ", " + str(storage["l2"]) +")$")

# plt.plot(xs,A[:,80],'k',label='True Solution')
# plt.plot(xs,C[:,-1],'r',label='POD Recon')
# plt.plot(xs,CR[:,-1],'g--',label='RandPOD Recon')
# plt.plot(xs,CS[:,-1],'b',label='SingleView Recon')
# plt.title("First 80 Columns")

# plt.subplot(1,3,3)

# plt.plot(xs,A[:,-1],'k',label='True Solution')
# plt.plot(xs,D[:,-1],'r',label='POD Recon')
# plt.plot(xs,DR[:,-1],'g--',label='RandPOD Recon')
# plt.plot(xs,DS[:,-1],'b',label='SingleView Recon')
# plt.title("All 120 Columns")


plt.legend(loc = 'lower right')
plt.tight_layout()
plt.savefig("Ma798_SingleViewROM/Figures/POD/reconstructions_mult" + str(mode_multiplier) + ".pdf")
plt.show()

