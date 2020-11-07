# This program aims to process the refraction seimic data. The first arrival 
# picking can be done with other program like obspy, sac and etc.
# Next, determin the cross-over distance of the direction and fraction wave and save 
# the first arrival of refraction wave as the format below:
#
# 		data format: Time series with only refraction point, and nodata should be 
#				     filled with -1. The first column is the off of the reciever.
#
# Other required inpput:
# 	1. shot distance/offset: In this case, because the source distance is not mohogeneous, 
#      the source offset of each "shot" should be given.
#   2. surface velocity v0
#
# smoothing:
#   The Laplacian matrix can also be used in the inversion. 
#    Uncomment the section can adjust the result.


import numpy as np 
import matplotlib.pyplot  as plt

v0=0.2
tij= np.loadtxt('layer2.csv', delimiter=',',skiprows=2)
source=np.array([0., 14., 24.,36., 48., 58. ,70.])

############ no need to change ##############

receiver=tij[:,0]
# inversion start
Ginit=[]
d=[]

plt.subplot(211)
plt.xlim(min(receiver), max(receiver))
for i in range(len(source)):
	mask=(tij[:,i+1]!=-1)
	plt.plot(receiver[mask],tij[:,i+1][mask],'o',ms=2)
	plt.ylim(0,70)
	for j  in range(len(receiver)):
		index=int((source[i]-12)/2)
		if i==0.:
			index=-2
		if i==len(source)-1:
			index=-1
		if tij[j,i+1] == -1:
			continue
		arr=[0]*(len(receiver)+2)
		arr[index]=1
		arr[j]=1
		arr.append(abs(source[i]-receiver[j]))
		Ginit.append(arr)
		d.append(tij[j,i+1])


Ginit=np.array(Ginit)

#smoothing
# for i in range(len(receiver)-1):
# 	lap=np.zeros(shape=len(receiver)+3)
# 	w=0
# 	lap[i]=1
# 	lap[i+1]=-1
# 	Ginit=np.vstack((Ginit,lap*w))
# 	d+=[0]
d=np.array(d)

# solve to inversion problem: m= inv(G*GT)*GT*d
invmat=np.linalg.solve(np.dot(Ginit.transpose(),Ginit), np.dot(Ginit.transpose(), d))

v=1/invmat[-1]
dcf=invmat[:-3]*v*v0/(v**2-v0**2)**0.5

## figure setting
plt.subplot(212)
plt.gca().invert_yaxis()
plt.fill_between(receiver ,dcf, 0 )
plt.fill_between(receiver ,dcf, max(dcf)+1 )
plt.xlim(min(receiver), max(receiver))
plt.ylim(max(dcf)+1,0)
plt.annotate("v=%.3f"%(v0), (receiver[-5],min(dcf-0.5)))
plt.annotate("v=%.3f"%(v), (receiver[-5],max(dcf)+0.5))
plt.show()