#%% writing testing 7 into a function
import numpy as np

def cubeAv_v3(layout,plot_points,theta_i,U_i,P_i,ct_f,cp_f,K,A,rho=1.225):
    #calculates the (average) wake velocity and farm aep discretely
    def deltaU_by_Uinf(r,theta,ct,K):
        theta = theta + np.pi #wake is downstream
        ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))

        U_delta_by_U_inf = (1-np.sqrt(1-(ct/(8*(K*r*np.sin(theta)+ep)**2))))*(np.exp(-(r*np.cos(theta))**2/(2*(K*r*np.sin(theta)+ep)**2)))

        lim = (np.sqrt(ct/8)-ep)/K #this is the y value of the invalid region, can be negative depending on Ct
        lim = np.where(lim>0,lim,0) #may sure it's always atleast 0
        deltaU_by_Uinf = np.where(r*np.sin(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit which can occur if lim<0
        if np.any( (r*np.sin(theta)>0.00001) & (r*np.sin(theta)<lim) ):
            print("turbines within invalid zone, careful")
        return deltaU_by_Uinf

    #I sometimes use this function to find the wake layout, so find relative posistions to plot points not the layout 
    #when plot_points = layout it finds wake at the turbine posistions
    xt_j,yt_j = layout[:,0],layout[:,1]
    xp_j,yp_j = plot_points[:,0],plot_points[:,1]

    x_jk = xp_j[:, None] - xt_j[None, :]
    y_jk = yp_j[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    theta_jk = np.arctan2(y_jk, x_jk)

    theta_ijk = theta_jk[None,:,:] + theta_i[:,None,None]
    r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)
    ct_ijk = ct_f(U_i)[...,None,None]*np.ones((len(xp_j),len(xt_j)))[None,...] #this is a dirty way of repeating along 2 axis
    Uw_ij = U_i[:,None]*(1-np.sum(deltaU_by_Uinf(r_ijk,theta_ijk,ct_ijk,K),axis=2))
    print("Uw_ij: {}".format(Uw_ij))
    if len(xt_j) == len(xp_j): #farm aep calculation
        pow_ij = 0.5*A*rho*cp_f(Uw_ij)*Uw_ij**3
        aep = np.sum(np.sum(pow_ij,axis=1)*P_i)/(1*10**6)
    else: #farm wake visualisation
        pow_ij = np.nan
        aep = np.nan
        Uw_ij = np.sum(P_i[:,None]*Uw_ij,axis=0)
    return Uw_ij,pow_ij,aep

xt = np.array((0,2,))
yt = np.array((0,2,))
layout = np.column_stack((xt,yt))

X,Y = np.meshgrid(np.linspace(-3,3,200),np.linspace(-3,3,200))
plot_grid = np.column_stack((X.reshape(-1),Y.reshape(-1)))

from AEP3_functions_v01 import y_5MW
turb = y_5MW()
#(0,np.pi/2,np.pi,3*np.pi/2)
theta_i = np.array((np.pi/4,)) #wind directions
U_i = np.array((10,))
P_i = np.array((1,))

A = np.pi * (126/2)**2  

U_w, pow, aep = cubeAv_v3(layout,layout,theta_i,U_i,P_i,turb.Ct_f(),turb.Cp_f(),0.06,A)
print(pow)
#%%
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
cf = ax.contourf(X,Y,U_w.reshape(X.shape),50,cmap=cm.coolwarm)
ax.scatter(layout[:,0],layout[:,1],s=30,marker='x',color='black')
fig.colorbar(cf)

#%%
a = np.array((1,2,3))
b = np.array((3,2,1))
c = (a==b) & (a==2)
