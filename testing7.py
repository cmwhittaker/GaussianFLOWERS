#%% This is an actually working version of the discrete numerical convolution
import numpy as np

xt_j = np.array((0,0.55,-0.55))
yt_j = np.array((0,-1.1,-1.1))

x_jk = xt_j[:, None] - xt_j
y_jk = yt_j[:, None] - yt_j

r_jk = np.sqrt(x_jk**2+y_jk**2)
theta_jk = np.arctan2(y_jk, x_jk)

theta_i = np.array((0,np.pi/2,np.pi,3*np.pi/2)) #wind directions
U_i = np.array((18,18,18,18))
P_i = np.array((1,1,1,1))

theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None]
r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)

def deltaU_by_Uinf(r,theta,ct,k):
    theta = theta + np.pi
    ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))

    U_delta_by_U_inf = (1-np.sqrt(1-(ct/(8*(k*r*np.sin(theta)+ep)**2))))*(np.exp(-(r*np.cos(theta))**2/(2*(k*r*np.sin(theta)+ep)**2)))

    lim = (np.sqrt(ct/8)-ep)/k #this is the y value of the invalid region, can be negative depending on Ct
    lim = np.where(lim>0,lim,0) #may sure it's always atleast 0
    deltaU_by_Uinf = np.where(r*np.sin(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit which can occur if lim<0
    return deltaU_by_Uinf

from AEP3_functions_v01 import y_5MW
turb = y_5MW()
ct_ijk = turb.Ct_itrp(U_i)[...,None,None]*np.ones((len(xt_j),len(xt_j)))[None,...] #this is a dirty way of repeating twice
k = 0.06
#wind speed for each direction
U_w = U_i[:,None]*(1-np.sum(deltaU_by_Uinf(r_ijk,theta_ijk,ct_ijk,k),axis=2))
A = np.pi * (126/2)**2  
pow = np.sum(0.5*A*1.225*turb.Cp_itrp(U_w)*U_w**3,axis=1)
print(pow/(1*10**6))
aep = np.sum(pow*P_i)/(1*10**6)

#%%
a = np.array((-1,0,1)) #the y limit
b = np.array((1,2,3))

a = np.where(a>0,a,0)
print(a)


#%%


import numpy as np

# Generate some example data
n = 5
xt_j = np.random.rand(n)
yt_j = np.random.rand(n)

# Compute the angle between each pair of points
dx = xt_j - xt_j[:, None]
dy = yt_j - yt_j[:, None]
angles = np.arctan2(dy, dx)
np.fill_diagonal(angles, np.nan) # set diagonal elements to NaN
angles = angles[:, ~np.isnan(angles).all(axis=0)] # remove columns with all NaNs

print(angles)