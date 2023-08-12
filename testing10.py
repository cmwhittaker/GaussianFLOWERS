#%% testing the re-written ntag function
import numpy as np

def ntag(layout,plot_points,cjd3_Fterms,CT,K,A,rho=1.225):
    #"No cross Term Analytical Gaussian" - the performance can probably be increased with a small amount of optimisation
    #CT is constant across all wind directions :(

    xt_j,yt_j = layout[:,0],layout[:,1]
    xp_j,yp_j = plot_points[:,0],plot_points[:,1]

    x_jk = xp_j[:, None] - xt_j[None, :]
    y_jk = yp_j[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    theta_jk = np.pi/2 - np.arctan2(y_jk, x_jk)
    #coord system conversion

    a_0,a_n,b_n = cjd3_Fterms

    EP = 0.2*np.sqrt((1+np.sqrt(1-CT))/(2*np.sqrt(1-CT)))

    #auxilaries
    n = np.arange(1,a_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    sqrt_term = np.where(r_jk<(np.sqrt(CT/8)-EP)/K,0,(1-np.sqrt(1-(CT/(8*(K*r_jk+EP)**2)))))

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]    
    sigma_b = sigma[:,:,None]
    a_n = a_n[None,None,:]
    b_n = b_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream

    def term(a):
        cnst_term = (np.sqrt(2*np.pi*a)*sigma/(2*a*np.pi))*(sqrt_term**a)
        mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2*a))*(a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)
    
    #alpha is the 'energy' content of the wind
    #I don't know why this 2pi is needed, *but it is*
    alpha = (a_0/2 - 3*term(1) + 3*term(2) - term(3))*2*np.pi
    if len(xt_j) == len(xp_j): #farm aep calculation
        pow_j = 0.5*A*rho*alpha
        aep = np.sum(pow_j)/(1*10**6)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan

    return alpha,pow_j,aep

xt = np.array((0,))
yt = np.array((0,))
layout = np.column_stack((xt,yt))

X,Y = np.meshgrid(np.linspace(-10,10,200),np.linspace(-10,10,200))
plot_grid = np.column_stack((X.reshape(-1),Y.reshape(-1)))

from AEP3_functions_v01 import y_5MW
turb = y_5MW()
from distributions_vC05 import wind_rose
wr = wind_rose(custom=2,a_0=8,Cp_f=turb.Cp_f())
fourier_coeffs = wr.cjd3_full_Fourier_coeffs

A = np.pi * (126/2)**2  
CT = turb.Cp_itrp(np.mean(wr.avMagnitude))
K = 0.06

alpha, pow_j, aep = ntag(layout,layout,fourier_coeffs,CT,K,A,rho=1.225)
print(pow_j)

#%%

import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
cf = ax.contourf(X,Y,alpha.reshape(X.shape),50,cmap=cm.coolwarm)
fig.colorbar(cf)
#

#%%
from AEP3_2_functions import y_5MW
turb = y_5MW()
a = turb.Cp_f
print(a(10))
#%%
import numpy as np
lim = 0.09
a = np.full((2,2),True)
b = np.full((2,2),False)
print(np.any(a & ~b))