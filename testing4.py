#%% for a single wind direction ... fixed Ct
import numpy as np

NT = 3
SPACING = 5

def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))

        lim = (np.sqrt(Ct/8)-ep)/K
        lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be <0 but there is numerical artifacting in rsin(theta) )
        lim = 3
        theta = theta + np.pi
        U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*np.cos(theta)+ep)**2))))*(np.exp(-(r*np.sin(theta))**2/(2*(K*r*np.cos(theta)+ep)**2)))
        deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit  
 
        return deltaU_by_Uinf 

def get_layout():
    xt = np.arange(-SPACING,SPACING+1,SPACING)
    yt = np.zeros_like(xt)
    return xt,yt,np.column_stack((xt,yt))

Ct = 0.8
K = 0.03

wind_direction = np.deg2rad(20)


#%% so this is done for a single wind direction, correct turbine ordering, and variable Ct
Xt,Yt,layout = get_layout()
from turbines_v01 import iea_10MW
turb = iea_10MW()
Ct_f = turb.Ct_f
theta_i = np.deg2rad(270)
U_i = 12
X,Y = np.meshgrid(np.linspace(-10,10,100),np.linspace(-10,10,100))
Uwff = U_i*np.ones_like(X)
Uwt = U_i*np.ones(layout.shape[0]) #turbine Uws

num_turbs = layout.shape[0]
Ct = Ct_f(U_i)
Cts = []
def get_sort_index(layout,rot):
    #rot:clockwise +ve
    Xt,Yt = layout[:,0],layout[:,1]
    rot_Xt = Xt * np.cos(rot) + Yt * np.sin(rot)
    rot_Yt = -Xt * np.sin(rot) + Yt * np.cos(rot) 
    layout = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
    sort_index = np.argsort(-layout[:, 1]) #sort index, with furthest upwind first
    return sort_index

sort_index = get_sort_index(layout,-theta_i)
layout = layout[sort_index]

for k in range(num_turbs): #for each turbine in superposistion
    xt, yt = layout[k,0],layout[k,1]      
    Rff = np.sqrt((X-xt)**2+(Y-yt)**2)
    THETAff = np.pi/2 - np.arctan2(Y-yt,X-xt) - theta_i

    Rt = np.sqrt((Xt-xt)**2+(Yt-yt)**2)
    THETAt = np.pi/2 - np.arctan2(Yt-yt,Xt-xt) - theta_i

    Ct = Ct_f(Uwt[k])
    Cts.append(Ct)
    Uwt = Uwt - U_i*deltaU_by_Uinf_f(Rt,THETAt,Ct,K)

    Uwff = Uwff - U_i*deltaU_by_Uinf_f(Rff,THETAff,Ct,K)

import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
Z = Uwff
#Z = np.where(Uwff<4.45,Uwff,np.NaN)
cf = ax.contourf(X,Y,Z,50,cmap=cm.coolwarm)
fig.colorbar(cf)
ax.set(aspect='equal')

for k in range(num_turbs): #show order of compute
     xt, yt = layout[k,0],layout[k,1]  
     ax.scatter(xt,yt,color='black')
     ax.annotate(str(k),(xt,yt))

#%% ok, so now it's done for a range of directions, with variable Ct
U_inf1 = 12
Ws = np.ones(2,)*U_inf1
Ps = np.ones(2,)*0.5 #two direction, equal chance
Wd = np.array((0,(3*np.pi)/2)) #wind direction

Xt,Yt,layout = get_layout()
from turbines_v01 import iea_10MW
turb = iea_10MW()
Ct_f = turb.Ct_f

X,Y = np.meshgrid(np.linspace(-10,10,100),np.linspace(-10,10,100))
og_shape = X.shape
X,Y = X.reshape(-1),Y.reshape(-1) #flatten arrays

Uwff = Ws[:,None]*np.ones(((1,X.shape[0])))
DUff = np.zeros((len(Ws),len(X),len(X)))

Uwt = Ws[:,None]*np.ones((1,Xt.shape[0])) #turbine Uws
DUt = np.zeros((len(Ws),len(Xt),len(Xt)))
    
num_turbs = layout.shape[0]

def get_sort_index(layout,rot):
    #rot:clockwise +ve
    Xt,Yt = layout[:,0],layout[:,1]
    rot_Xt = Xt * np.cos(rot) + Yt * np.sin(rot)
    rot_Yt = -Xt * np.sin(rot) + Yt * np.cos(rot) 
    layout = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
    sort_index = np.argsort(-layout[:, 1]) #sort index, with furthest upwind first
    return sort_index

#the actual calculation loop
for i in range(len(Ws)): #for each wind direction
    
    sort_index = get_sort_index(layout,-Wd[i]) #find
    layout = layout[sort_index] #and reorder based on furthest upwind

    for k in range(len(Xt)): #for each turbine in superposistion
        xt, yt = layout[k,0],layout[k,1]       
        #turbine locations 
        Rt = np.sqrt((Xt-xt)**2+(Yt-yt)**2)
        THETAt = np.pi/2 - np.arctan2(Yt-yt,Xt-xt) - Wd[i]
        #these are the flow field mesh
        Rff = np.sqrt((X-xt)**2+(Y-yt)**2)
        THETAff = np.pi/2 - np.arctan2(Y-yt,X-xt) - Wd[i]

        Ct = Ct_f(Uwt[i,k])
        DUt[i,:,k] = deltaU_by_Uinf_f(Rt,THETAt,Ct,K)
        Uwt[i,:] = Uwt[i,:] - Ws[i]*DUt[i,:,k]
        
        DUff[i,:,k] = deltaU_by_Uinf_f(Rff,THETAff,Ct,K)
        Uwff[i,:] = Uwff[i,:] - Ws[i]*DUff[i,:,k]

import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
Z = Uwff[1,:]
X,Y,Z = [a.reshape(og_shape) for a in [X,Y,Z]]
#Z = np.where(Uwff<4.45,Uwff,np.NaN)
cf = ax.contourf(X,Y,Z,50,cmap=cm.coolwarm)
fig.colorbar(cf)
ax.set(aspect='equal')

for k in range(num_turbs): #show order of compute
     xt, yt = layout[k,0],layout[k,1]  
     ax.scatter(xt,yt,color='black')
     ax.annotate(str(k),(xt,yt))


#%% ok, that's great! now package it nicely in a function and add back all the aep functionality.

def get_layout(nt,spacing):
    xt = np.arange(-nt//2*spacing,nt//2*spacing+1,spacing)
    yt = np.zeros_like(xt)
    return xt,yt,np.column_stack((xt,yt))

import numpy as np
def num_F_v02(U_i,P_i,theta_i,
              layout,
              plot_points, #this is the comp domain
              turb,
              RHO=1.225,K=0.025,
              u_lim=None,Ct_op=True,cross_ts=True,ex=True,lcl_Cp=True,avCube=True):
    
    def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
        if u_lim is not None:
            lim = u_lim
        else:
            lim = (np.sqrt(Ct/8)-ep)/K
            lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) 
        
        theta = theta + np.pi #the wake lies opposite!
        if ex: #use full 
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*np.cos(theta)+ep)**2))))*(np.exp(-(r*np.sin(theta))**2/(2*(K*r*np.cos(theta)+ep)**2)))
            deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit  
        else: #otherwise use small angle approximations
            theta = np.mod(theta-np.pi,2*np.pi)-np.pi
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*1+ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+ep)**2)))          
            deltaU_by_Uinf = np.where(r>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
            return deltaU_by_Uinf      
        
        return deltaU_by_Uinf  
    
    def get_sort_index(layout,rot):
        #rot:clockwise +ve
        Xt,Yt = layout[:,0],layout[:,1]
        rot_Xt = Xt * np.cos(rot) + Yt * np.sin(rot)
        rot_Yt = -Xt * np.sin(rot) + Yt * np.cos(rot) 
        layout = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
        sort_index = np.argsort(-layout[:, 1]) #sort index, with furthest upwind first
        return sort_index
    
    def soat(a): #Sum over Axis Two
        return np.sum(a,axis=2)

    Xt,Yt = layout[:,0],layout[:,1]
    X,Y = plot_points[:,0],plot_points[:,1] #flatten arrays

    WAV_CT = np.sum(Ct_f(U_i)*P_i) #the weight-averaged Ct
    
    DUt_ijk = np.zeros((len(U_i),len(Xt),len(Xt)))
    Uwt_ij = U_i[:,None]*np.ones((1,Xt.shape[0])) #turbine Uws

    DUff_ijk = np.zeros((len(U_i),len(X),len(Xt)))
    Uwff_ij = U_i[:,None]*np.ones(((1,X.shape[0])))

    #the actual calculation loop
    for i in range(len(U_i)): #for each wind direction
        
        sort_index = get_sort_index(layout,-theta_i[i]) #find
        layout = layout[sort_index] #and reorder based on furthest upwind

        for k in range(len(Xt)): #for each turbine in superposistion
            xt, yt = layout[k,0],layout[k,1]       
            #turbine locations 
            Rt = np.sqrt((Xt-xt)**2+(Yt-yt)**2)
            THETAt = np.pi/2 - np.arctan2(Yt-yt,Xt-xt) - theta_i[i]
            #these are the flow field mesh
            Rff = np.sqrt((X-xt)**2+(Y-yt)**2)
            THETAff = np.pi/2 - np.arctan2(Y-yt,X-xt) - theta_i[i]

            if Ct_op == 1: # (1 or True care!) use local Ct
                Ct = turb.Ct_f(Uwt_ij[i,k]) #base on local wake velocity
            elif Ct_op == 2:
                Ct = turb.Ct_f(U_i[i]) #base on global wake velocity
            elif Ct_op == 3:
                Ct = WAV_CT #weight-averaged Ct
            else:
                raise ValueError("Ct_op should be 1,2 or 3")

            DUt_ijk[i,:,k] = deltaU_by_Uinf_f(Rt,THETAt,Ct,K)
            Uwt_ij[i,:] = Uwt_ij[i,:] - U_i[i]*DUt_ijk[i,:,k] #sum over k
            
            DUff_ijk[i,:,k] = deltaU_by_Uinf_f(Rff,THETAff,Ct,K)
            Uwff_ij[i,:] = Uwff_ij[i,:] - U_i[i]*DUff_ijk[i,:,k] #sum over k
    
    #calculate power at the turbine location
    if cross_ts: #INcluding cross terms
        Uwt_ij_cube = Uwt_ij**3
    else: #EXcluding cross terms (soat = Sum over Axis Two (third axis!)
        Uwt_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(DUt_ijk) + 3*soat(DUt_ijk**2) - soat(DUt_ijk**3))

    if lcl_Cp: #power coeff based on local wake velocity
        Cp_ij = turb.Cp_f(Uwt_ij)
    else: #power coeff based on global inflow U_infty
        Cp_ij = turb.Cp_f(U_i)[:,None]

    #sum over wind directions (i) (this is the weight-averaging)
    if avCube: #directly find the average of the cube velocity
        pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uwt_ij_cube),axis=0)/(1*10**6)
    else: #the old way of cubing the weight-averaged field
        WAV_CP = np.sum(turb.Cp_f(U_i)*P_i) #frequency-weighted av Cp on global
        pow_j = 0.5*turb.A*RHO*WAV_CP*np.sum(P_i[:,None]*Uwt_ij**3,axis=0)/(1*10**6)
    #(j in Uwff_j here is indexing the meshgrid)
    Uwff_j = np.sum(P_i[:,None]*Uwff_ij,axis=0) #weighted flow field
    
    return pow_j,Uwff_j 

NT = 3
SPACING = 7
Xt,Yt,layout = get_layout(NT,SPACING)
from turbines_v01 import iea_10MW
turb = iea_10MW()

# U_inf1 = 20
# U_i = np.ones(2,)*U_inf1
# P_i = np.ones(2,)*0.5 #two direction, equal chance
# theta_i = np.array((0,(3*np.pi)/2)) #wind direction

U_inf1 = 10
U_i = np.ones(1,)*U_inf1
P_i = np.ones(1,) #two direction, equal chance
theta_i = np.array(((3*np.pi)/2,)) #wind direction

X,Y = np.meshgrid(np.linspace(-20,20,300),np.linspace(-10,10,300))
og_shape = X.shape
X,Y = X.reshape(-1),Y.reshape(-1) #flatten arrays
plot_points = np.column_stack((X.reshape(-1),Y.reshape(-1)))

a1,b1 = num_F_v02(U_i,P_i,theta_i,
              layout,
              plot_points, #this is the comp domain
              turb,
              RHO=1.225,K=0.025,
              u_lim=None,Ct_op=1,cross_ts=True,ex=True,lcl_Cp=True,avCube=True)

np.sum(pow)

from AEP3_3_functions import num_F,gen_local_grid_v01C
r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)

a2,b2 = num_F(U_i,P_i,theta_i,
          r_jk,theta_jk,
          turb,
          RHO=1.225,K=0.025,
          u_lim=None,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)

print(a1)
print(a2)
#%%

import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
Z = b1
X,Y,Z = [a.reshape(og_shape) for a in [X,Y,Z]]
#Z = np.where(Uwff<4.45,Uwff,np.NaN)
cf = ax.contourf(X,Y,Z,50,cmap=cm.coolwarm)
fig.colorbar(cf)
#ax.set(aspect='equal')

for k in range(len(Xt)): #show order of compute
     xt, yt = layout[k,0],layout[k,1]  
     ax.scatter(xt,yt,color='black')
     ax.annotate(str(k+1),(xt,yt))
