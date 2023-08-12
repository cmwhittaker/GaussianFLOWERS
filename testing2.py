#%% set font!
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

#%%generate the data
import numpy as np

NT = 5 #number of turbines 
SPACING = 7 
XPAD = 7 #X pad
YPAD = 7
BINS = 360
U_inf1 = 10 #12 or 5
U_inf2 = 12
U_LIM = 4

CP_LOWER = 0
CP_UPPER = 27

SAVE_FIG = False

EXTENT = 14

RES = 201

def get_layout():
    xt = np.arange(0,NT*SPACING,SPACING)
    yt = np.zeros_like(xt)
    return xt,yt,np.column_stack((xt,yt))

def rectangular_domain(r=RES):
    xx,yy = np.meshgrid(np.linspace(-XPAD,(NT-1)*SPACING+XPAD,r),np.linspace(-YPAD,YPAD,r))
    return xx,yy,np.column_stack((xx.reshape(-1),yy.reshape(-1)))

def get_real_wr():
    from distributions_vC05 import wind_rose
    wr = wind_rose(bin_no_bins=BINS,custom=None,site=6,Cp_f=turb.Cp_f)
    U_i = wr.avMagnitude
    P_i = wr.frequency
    return U_i,P_i

def custom_wr():
    U_i = U_inf1*np.ones(BINS)
    from scipy.stats import vonmises
    kappa = 8.0  # concentration parameter
    mu1 = (3/2)*np.pi # mean direction (in radians)
    P_i = vonmises.pdf(theta_i, kappa, loc=mu1)
    P_i = P_i/np.sum(P_i) #normalise for discrete distribution
    return U_i,P_i 

from turbines_v01 import iea_10MW
turb = iea_10MW()
Ct_f = turb.Ct_f
Cp_f = turb.Cp_f

U_i = np.ones(360,)*U_inf1
P_i = np.zeros_like(U_i)
P_i[270] = 1 #blows from a single wind direction
theta_i = np.linspace(0,2*np.pi,BINS,endpoint=False)
#U_i,P_i = get_real_wr()
U_i,P_i = custom_wr()

xt,yt,layout = get_layout()
xx,yy,plot_points = rectangular_domain()

global_var = 1

from AEP3_3_functions import gen_local_grid_v01C
#slightly modified numerical function
def num_F_v01(U_i,P_i,theta_i,
          layout,
          turb,
          RHO=1.225,K=0.025,
          u_lim=None,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True):
    #function to show the different effects of the many assumptions
    #i:directions,j:turbines,k:turbines in superposistion
    #invalid: specific an invalid radius
    #cross_t: cross terms in cubic expansion
    #sml_a: small_angle approximation
    #local_cp:local power coeff (or global)
    #(var_ct: ct is fixed externally with a lambda function if wanted)

    def gen_local_grid(layout,plot_points):
        xt_j,yt_j = layout[:,0],layout[:,1]
        print(len(xt_j))
        xt_k,yt_k = plot_points[:,0],plot_points[:,1]

        x_jk = xt_k[:, None] - xt_j[None, :]
        y_jk = yt_k[:, None] - yt_j[None, :]

        r_jk = np.sqrt(x_jk**2+y_jk**2)
        #convert theta from clckwise -ve x axis to clckwise +ve y axis 
        theta_jk = np.pi/2 - np.arctan2(y_jk, x_jk)

        return r_jk,theta_jk

    def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
        
        if u_lim is not None:
            lim = u_lim
        else:
            lim = (np.sqrt(Ct/8)-ep)/K
            lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be <0 but there is numerical artifacting in rsin(theta) )
        
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

    Ct_f = turb.Ct_f
    Cp_f = turb.Cp_f
    A = turb.A

    if not var_Ct: #use the Fixed (Weight averaged) Ct?     
        WAV_CT = np.sum(Ct_f(U_i)*P_i)
        Ct_f = lambda x: WAV_CT

    r_jk,theta_jk = gen_local_grid(layout,layout)

    #when plot_points == layout it finds wake at the turbine posistions
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None]
    
    r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)
    ct_ijk = Ct_f(U_i)[...,None,None]*np.ones((r_jk.shape[0],r_jk.shape[1]))
    [None,...] #this is a dirty way of repeating along 2 axis
        
    def soat(a): #Sum over Axis Two
        return np.sum(a,axis=2)

    DU_by_Uinf_ijk = deltaU_by_Uinf_f(r_ijk,theta_ijk,ct_ijk,K) #deltaU_by_Uinf as a function
    if cross_ts: #INcluding cross terms
        Uw_ij_cube = (U_i[:,None]*(1-np.sum(DU_by_Uinf_ijk,axis=2)))**3
    else: #EXcluding cross terms (soat = Sum over Axis Two (third axis!)
        Uw_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(DU_by_Uinf_ijk) + 3*soat(DU_by_Uinf_ijk**2) - soat(DU_by_Uinf_ijk**3))

    Uw_ij = (U_i[:,None]*(1-np.sum(DU_by_Uinf_ijk,axis=2)))
    if lcl_Cp: #power coeff based on local wake velocity
        Cp_ij = Cp_f(Uw_ij)
    else: #power coeff based on global inflow U_infty
        Cp_ij = Cp_f(U_i)[:,None]

    #sum over wind directions (i) (this is the weight-averaging)
    if avCube: #directly find the average of the cube velocity
        pow_j = 0.5*A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uw_ij_cube),axis=0)/(1*10**6)
    else: #the old way of cubing the weight-averaged field
        WAV_CP = np.sum(Cp_f(U_i)*P_i) #frequency-weighted av Cp on global
        pow_j = 0.5*A*RHO*WAV_CP*np.sum(P_i[:,None]*Uw_ij**3,axis=0)/(1*10**6)

    Uw_j = np.sum(P_i[:,None]*Uw_ij,axis=0) #flow field
    return pow_j,Uw_j #power(mw)/wake velocity 

r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
#aep first
global_var
a1,b1 = num_F_v01(U_i,P_i,theta_i,
                 layout,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)
#that is the vectorised way
print(np.sum(a1))

#%% a partly non-vectorised way?

def find_ct_in_turn():
    for i in range(len(U_i)): #for each wind direction
        rot_layout = rotate(layout,-theta_i[i])
        sort_index = np.argsort(-rot_layout[:, 1]) #sort with the greatest y value first (most upwind)
        rot_layout = rot_layout[sort_index]
        for j in range(layout.shape[0]): #for each turbine
            #now go through and add velocity deficits for each turbine
            deltaU_by_Uinf += deltaU_by_Uinf_f()
            pass
    return None

#%% non-vectorised way

def rotate(layout,rot):
    
    Xt,Yt = layout[:,0],layout[:,1]

    rot_Xt = Xt * np.cos(phi) - Yt * np.sin(phi)
    rot_Yt = Xt * np.sin(phi) + Yt * np.cos(phi)

    return np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))

foo = rotate_points(layout,10)
xt,yt = foo[:,0],foo[:,1]
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.scatter(xt,yt)
ax.set(aspect='equal')

#%%



#%%
#probably far easier to do it for single direction first

U_w = np.ones(5,5)

for j in range(layout.shape[0]): #for each turbine
    pass
#%%
u_lim = None
ex = True
def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
        
        if u_lim is not None:
            lim = u_lim
        else:
            lim = (np.sqrt(Ct/8)-ep)/K
            lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be <0 but there is numerical artifacting in rsin(theta) )
        
        theta = theta + np.pi #the wake lies opposite!
        if ex: #use full 
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*np.cos(theta)+ep)**2))))*(np.exp(-(r*np.sin(theta))**2/(2*(K*r*np.cos(theta)+ep)**2)))
            deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit  
        else: #otherwise use small angle approximations
            theta = np.mod(theta-np.pi,2*np.pi)-np.pi
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*1+ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+ep)**2)))          
            deltaU_by_Uinf = np.where(r>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
            return deltaU_by_Uinf
        
#probably easier to do it for a SINGLE turbine
layout = np.array(((0,0),),)
plot_points = rectangular_domain(100) 

#%%
import numpy as np

def num_F_v01(U_i, P_i, theta_i,
              r_jk, theta_jk,
              turb,
              RHO=1.225, K=0.025,
              u_lim=None, cross_ts=True, ex=True, lcl_Cp=True, avCube=True, var_Ct=True):

    def deltaU_by_Uinf_f(r, theta, Ct, K, u_lim, ex):
        ep = 0.2 * np.sqrt((1 + np.sqrt(1 - Ct)) / (2 * np.sqrt(1 - Ct)))

        if u_lim is not None:
            lim = u_lim
        else:
            lim = (np.sqrt(Ct / 8) - ep) / K
            lim = max(lim, 0.01)
        
        theta = theta + np.pi
        if ex:
            U_delta_by_U_inf = (1 - np.sqrt(1 - (Ct / (8 * (K * r * np.cos(theta) + ep) ** 2)))) * (
                    np.exp(-(r * np.sin(theta)) ** 2 / (2 * (K * r * np.cos(theta) + ep) ** 2)))
            deltaU_by_Uinf = U_delta_by_U_inf if r * np.cos(theta) > lim else 0
        else:
            theta = np.mod(theta - np.pi, 2 * np.pi) - np.pi
            U_delta_by_U_inf = (1 - np.sqrt(1 - (Ct / (8 * (K * r * 1 + ep) ** 2)))) * (
                    np.exp(-(r * theta) ** 2 / (2 * (K * r * 1 + ep) ** 2)))
            deltaU_by_Uinf = U_delta_by_U_inf if r > lim else 0
        return deltaU_by_Uinf

    Ct_f = turb.Ct_f
    Cp_f = turb.Cp_f
    A = turb.A

    if not var_Ct:
        WAV_CT = np.sum(Ct_f(U_i) * P_i)
        Ct_f = lambda x: WAV_CT

    ni = len(U_i)
    nj, nk = r_jk.shape

    DU_by_Uinf_ijk = np.empty((ni, nj, nk))
    Uw_ij_cube = np.empty((ni, nj))
    Uw_ij = np.empty((ni, nj))
    Cp_ij = np.empty((ni, nj))

    for i in range(ni):
        for j in range(nj):
            if j==0:

            ct_ijk = Ct_f(U_i[i])
            ct_ijk = Ct_f(Uw_ij[i, j])
                
            for k in range(nk):
                theta_ijk = theta_jk[j, k] - theta_i[i]
                r_ijk = r_jk[j, k]
                

                DU_by_Uinf_ijk[i, j, k] = deltaU_by_Uinf_f(r_ijk, theta_ijk, ct_ijk, K, u_lim, ex)

            sum_DU_by_Uinf = np.sum(DU_by_Uinf_ijk[i, j, :])

            if cross_ts:
                Uw_ij_cube[i, j] = (U_i[i] * (1 - sum_DU_by_Uinf)) ** 3
            else:
                Uw_ij_cube[i, j] = (U_i[i] ** 3) * (1 - 3 * sum_DU_by_Uinf + 3 * sum_DU_by_Uinf ** 2 - sum_DU_by_Uinf ** 3)

            Uw_ij[i, j] = U_i[i] * (1 - sum_DU_by_Uinf)

            if lcl_Cp:
                Cp_ij[i, j] = Cp_f(Uw_ij[i, j])
            else:
                Cp_ij[i, j] = Cp_f(U_i[i])

    pow_j = np.empty(nj)
    Uw_j = np.empty(nj)

    for j in range(nj):
        if avCube:
            pow_j[j] = 0.5 * A * RHO * np.sum(P_i * (Cp_ij[:, j] * Uw_ij_cube[:, j])) / (1 * 10 ** 6)
        else:
            WAV_CP = np.sum(Cp_f(U_i) * P_i)
            pow_j[j] = 0.5 * A * RHO * WAV_CP * np.sum(P_i * Uw_ij[:, j] ** 3) / (1 * 10 ** 6)

        Uw_j[j] = np.sum(P_i * Uw_ij[:, j])
    print("indexy function")
    return pow_j, Uw_j

r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
#aep first
global_var
a2,b2 = num_F_v01(U_i,P_i,theta_i,
                 r_jk,theta_jk,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)
