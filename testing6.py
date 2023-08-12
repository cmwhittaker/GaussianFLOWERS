#%%writing a decent function for a change (LOL)
import numpy as np

def U_wake(xt,yt,U_infty,ct,k):
    ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))
    #nicely numpy correctly broadcasts (m,n,1)*(n)->(m,n,n)
    x_k = xt[...,None]-xt[None,:] 
    y_k = yt[...,None]-yt[None,:]
    r_k = np.sqrt(x_k**2+y_k**2)
    theta_k = np.arctan2(y_k,x_k) + np.pi   

    U_delta_by_U_inf = (1-np.sqrt(1-(ct[...,None]/(8*(k*r_k*np.cos(theta_k)+ep[...,None])**2))))*(np.exp(-(r_k*np.sin(theta_k))**2/(2*(k*r_k*np.cos(theta_k)+ep[...,None])**2)))

    lim = (np.sqrt(ct/8)-ep)/k #this is the x value of the invalid region, can be negative depending on Ct
    lim = np.where(lim>0,lim,0) #may sure it's always atleast 0
    U_delta_by_U_inf = np.where(r_k*np.cos(theta_k)>=lim[...,None],U_delta_by_U_inf,0) #this stops turbines producing their own deficit which can occur if lim<0
    
    return U_infty*(1-np.sum(U_delta_by_U_inf,axis=-1))

from AEP3_functions_v01 import y_5MW
turbine = y_5MW() #the turbine
Cp_f = turbine.Cp_f()
Ct_f = turbine.Ct_f()
xt = np.array((-3,0,-3))
xt = np.tile(xt[:,None],3).T
yt = np.array((-0.2,0,0.2))
yt = np.tile(yt[:,None],3).T

U_infty = np.full_like(xt,3)
ct = Ct_f(U_infty)
k=0.03

alpha = U_wake(xt,yt,U_infty,ct,k)
alpha

#%% now do it for multiple wind directions
wind_directions = np.linspace(0,2*np.pi,72,endpoint=False)

a = np.array([1,2,3])
b = np.array([4,5])
c = a[:,None]*b[None,:]

d = c[...,None]-c[...,:].T
#%%
a = xt[...,:].T
