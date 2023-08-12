#%%
import numpy as np

NT = 3 #number of turbines 
SPACING = 5 
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

U_i = np.ones(2,)*U_inf1
P_i = np.ones(2,)*0.5
theta_i = np.array((0,np.pi/2))

# P_i = np.zeros_like(U_i)
#P_i[270] = 1 #blows from a single wind direction
#theta_i = np.linspace(0,2*np.pi,BINS,endpoint=False)
#U_i,P_i = get_real_wr()


xt,yt,layout = get_layout()
xx,yy,plot_points = rectangular_domain()
#%%
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.scatter(xt,yt)
#%% non-vectorised calculations

import numpy as np

def num_F_v01(U_i, P_i, theta_i, layout, turb, RHO=1.225, K=0.025,
              u_lim=None, cross_ts=True, ex=True, lcl_Cp=True, avCube=True, var_Ct=True):

    def deltaU_by_Uinf_f(r, theta, Ct, K):
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
            theta = (theta - np.pi) % (2 * np.pi) - np.pi
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

    num_directions = len(U_i)
    num_turbines = layout.shape[0]

    DU_by_Uinf_ijk = np.zeros((num_directions, num_turbines, num_turbines))
    Uw_ij_cube = np.zeros((num_directions, num_turbines))

    Uf_ij = np.ones((num_directions,num_turbines))
    ct_ij = np.zeros((num_directions, num_turbines))

    for i in range(num_directions): #for each wind direction
        for j in range(num_turbines): #for each turbine
            xt_j, yt_j = layout[j, 0], layout[j, 1]
            
            # Rotate coordinates and store them
            rotated_coords = []
            for k in range(num_turbines):
                # Rotate the coordinates
                rotation_angle = -theta_i[i]
                cos_theta = np.cos(rotation_angle)
                sin_theta = np.sin(rotation_angle)
                x_rotated = (layout[k, 0] - xt_j) * cos_theta - (layout[k, 1] - yt_j) * sin_theta
                y_rotated = (layout[k, 0] - xt_j) * sin_theta + (layout[k, 1] - yt_j) * cos_theta
                rotated_coords.append((x_rotated, y_rotated))
            
            # Find most upwind turbine and reorder
            rotated_coords.sort(key=lambda coord: coord[1], reverse=True)
            
            # add superposistion of turbines
            for k, (x_rotated, y_rotated) in enumerate(rotated_coords):
                # For each turbine in superposistion
                r_jk = np.sqrt(x_rotated ** 2 + y_rotated ** 2)
                theta_jk = np.pi / 2 - np.arctan2(y_rotated, x_rotated)
                
                theta_ijk = theta_jk
                ct_ij[i,j] = Ct_f(Uf_ij[i, j]*U_i[i])
                DU_by_Uinf_ijk[i, j, k] = deltaU_by_Uinf_f(r_jk, theta_ijk, ct_ij[i,j], K)
                
                Uf_ij[i, j] = Uf_ij[i, j] - DU_by_Uinf_ijk[i, j, k]
            

            if cross_ts:
                Uw_ij_cube[i, j] = (U_i[i] * (1 - np.sum(DU_by_Uinf_ijk[i, j, :]))) ** 3
            else:
                Uw_ij_cube[i, j] = (U_i[i] ** 3) * (1 - 3 * np.sum(DU_by_Uinf_ijk[i, j, :]) +
                                                    3 * np.sum(DU_by_Uinf_ijk[i, j, :] ** 2) -
                                                    np.sum(DU_by_Uinf_ijk[i, j, :] ** 3))
    
    Uw_ij = U_i[i] * (1 - np.sum(DU_by_Uinf_ijk[i, j, :]))
    
    if avCube:
        pow_j = 0.5 * A * RHO * np.sum(P_i[:, None] * (Cp_f(Uw_ij) * Uw_ij_cube), axis=0) / (1 * 10 ** 6)
    else:
        WAV_CP = np.sum(Cp_f(U_i) * P_i)
        pow_j = 0.5 * A * RHO * WAV_CP * np.sum(P_i[:, None] * Uw_ij ** 3, axis=0) / (1 * 10 ** 6)

    Uw_j = np.sum(P_i[:, None] * Uw_ij, axis=0)
    return pow_j, Uw_j,ct_ij

a1,b1,c1 = num_F_v01(U_i,P_i,theta_i,
                 layout,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)
c1