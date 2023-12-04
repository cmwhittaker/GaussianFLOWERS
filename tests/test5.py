#%% Checking the Jensen FLOWERS implementation

# 7 shaped layout with a numerical equivalent of Jensen FLOWERS and the Jensen FLOWERS method
# (checking they match)
# subject to
# U_i = [15,13]
# P_i = [0.7,0.3]
# theta_i = [0,np.pi/2]


import sys
import os
sys.path.append(os.path.join('..', 'src')) #allow to import from utilities (there may be a better way ...)

if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2
import numpy as np
from utilities.turbines import iea_10MW
turb = iea_10MW()

K=0.05 #wake expansion rate
layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))

U_inf1 = 15 #15
U_inf2 = 13
U_WB_i = np.array((U_inf1,U_inf2,))
P_WB_i = np.array((0.7,0.3,))
theta_WB_i = np.array((0,np.pi/2,))

def U_delta_J(U_i,x,y):
    # U delta Jesen (cartesian)
    Ct = turb.Ct_f(U_i) #base thrust coefficent on free stream
    du = ((1-np.sqrt(1-Ct))/(2*K*x+1)**2) # 
    w = np.where((np.abs(y)<K*x+1/2)&(x>=0),1,0)  
    return du*w

def U_delta_J2(U_i,x,y):
    # U delta Jesen using 2nd Order Taylor approximation
    # using CARTESIAN coordinates
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x) 
    #second order taylor series approximation of the Jesen wake model
    Ct = turb.Ct_f(U_i) #base thrust coefficent on free stream
    du = (1-np.sqrt(1-Ct))*(1/(2*K*r+1)**2+(2*K*r*theta**2)/(2*K*r+1)**3)
    theta_c = np.arctan(
            (1 / (2*r) + K * np.sqrt(1 + K**2 - (2*r)**(-2)))
            / (-K / (2*r) + np.sqrt(1 + K**2 - (2*r)**(-2)))
            ) 
    w = np.where((-theta_c<=theta)&(theta<=theta_c),1,0)
    return du*w

def U_delta_J3(U_i,r,theta,K):
    # U delta Jesen using 2nd Order Taylor approximation
    # using POLAR coordinates
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
    #second order taylor series approximation of the Jesen wake model
    Ct = turb.Ct_f(U_i) #base thrust coefficent on free stream
    du = (1-np.sqrt(1-Ct))*(1/(2*K*r+1)**2+(2*K*r*theta**2)/(2*K*r+1)**3)
    theta_c = np.arctan2((1 / (2*r) + K * np.sqrt(1 + K**2 - (2*r)**(-2))),(-K / (2*r) + np.sqrt(1 + K**2 - (2*r)**(-2)))) 
    w = np.where((-theta_c<=theta)&(theta<=theta_c),1,0)
    return du*w

# === North @ U_inf1 ===
#T0, T1 are unwaked
DU_nT0T1 = 0
#T2 is waked by T1
DU_nT2 = U_delta_J2(U_WB_i[0],3,0.2)
#T3 is waked by T1 and T2
DU_nT3 = U_delta_J2(U_WB_i[0],6,0.4)+U_delta_J2(U_WB_i[0],3,0.2)
#wa velocity deficit for the North direction 
DU_wav_n = U_WB_i[0]*P_WB_i[0]*np.array((DU_nT0T1,DU_nT0T1,DU_nT2,DU_nT3))

# === East @ U_inf2 ===
#T1,T2,T3 are unwaked
DU_eT1T2T3 = 0
#T0 is waked by T1
DU_eT0 = U_delta_J2(U_WB_i[1],3,0)
#wa velocity deficit for the East direction 
DU_wav_e = U_WB_i[1]*P_WB_i[1]*np.array((DU_eT1T2T3,DU_eT1T2T3,DU_eT1T2T3,DU_eT0))

# === Totals ===
#weight-averaged total deficit
Ui_wav = np.sum(U_WB_i*P_WB_i) #weight average velocity
Uw_wav_t = Ui_wav - (DU_wav_n+DU_wav_e)
#power from weight-averaged velocities
alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst
simple_aep = np.sum(alpha*turb.Cp_f(Uw_wav_t)*Uw_wav_t**3)

from utilities.helpers import find_relative_coords,trans_bearing_to_polar,get_floris_wind_rose

def JF_num(U_i,P_i,theta_i,layout,turb,K,RHO=1.225):
    #numerical convolution using the 2nd order taylor approximation of the Jesen wake model
    r_jk,theta_jk = find_relative_coords(layout,layout)
    U_wav = np.sum(U_i*P_i)
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None]
    r_ijk =  np.broadcast_to(r_jk[None,:,:],theta_ijk.shape)
    U_ijk = np.broadcast_to(U_i[:,None,None],theta_ijk.shape)
    P_ijk = np.broadcast_to(P_i[:,None,None],theta_ijk.shape)

    DU_wav = np.sum(U_ijk*P_ijk*(U_delta_J3(U_ijk,r_ijk,theta_ijk,K)),axis=(0,2))
    U_w_wav = U_wav - DU_wav
    alpha = ((0.5*RHO*turb.A)/(1*10**6)) #turbine cnst
    simple_aep2 = np.sum(alpha*turb.Cp_f(U_w_wav)*U_w_wav**3)
    return simple_aep2
    
from utilities.helpers import find_relative_coords
def flowers_2(Fourier_coeffs1,
              layout1,layout2,
              turb,
              K,
              c_0,
              RHO=1.225):
    
    R,THETA = find_relative_coords(layout1,layout2)

    a_0, a_n, b_n = Fourier_coeffs1 #unpack
    
    # Set up mask for rotor swept area
    mask_area = np.where(R<=0.5,1,0) #true within radius

    # Critical polar angle of wake edge (as a function of distance from turbine)
    theta_c = np.arctan(
        (1 / (2*R) + K * np.sqrt(1 + K**2 - (2*R)**(-2)))
        / (-K / (2*R) + np.sqrt(1 + K**2 - (2*R)**(-2)))
        ) 
    theta_c = np.nan_to_num(theta_c)
    
    # Contribution from zero-frequency Fourier mode
    du = a_0 * theta_c / (2 * K * R + 1)**2 * (
        1 + (2*(theta_c)**2 * K * R) / (3 * (2 * K * R + 1)))
    
    # Reshape variables for vectorized calculations
    m = np.arange(1, len(a_n)+1) #half open interval
    a = a_n[None, None,:] 
    b = b_n[None, None,:] 
    R = R[:, :, None]
    THETA = THETA[:, :, None] 
    theta_c = theta_c[:, :, None] 

    # Vectorized contribution of higher Fourier modes
    du += np.sum(
        (2*(a * np.cos(m*THETA) + b * np.sin(m*THETA)) / (m * (2 * K * R + 1))**3 *
        (
        np.sin(m*theta_c)*(m**2*(2*K*R*(theta_c**2+1)+1)-4*K*R)+ 4*K*R*theta_c*m *np.cos(theta_c * m))
        ), axis=2)
    # Apply mask for points within rotor radius
    du = np.where(mask_area,a_0,du)
    np.fill_diagonal(du, 0.) #stop self-produced wakes (?)
    # Sum power for each turbine
    du = np.sum(du, axis=1) #superposistion sum
    wav = (c_0*np.pi - du)
    alpha = turb.Cp_f(wav)*wav**3 
    aep = (0.5*turb.A*RHO*alpha)/(1*10**6)

    return aep

import warnings
# Suppress all runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def new_wr1(no_bins):
    # ntag needs a large number of bins so that the spikes in 
    # the Fourier series are "sufficiently" narrow 
    if not no_bins%4 == 0:
        raise ValueError("no_bins must be a multiple of 4") 
    U_WB_i2 = np.zeros(no_bins)
    U_WB_i2[0],U_WB_i2[no_bins//4] = U_WB_i[0], U_WB_i[1]
    P_WB_i2 = np.zeros(no_bins)
    P_WB_i2[0],P_WB_i2[no_bins//4] = P_WB_i[0], P_WB_i[1]
    theta_WB_i2 = np.linspace(0,2*np.pi,no_bins,endpoint=False)
    return U_WB_i2, P_WB_i2, theta_WB_i2

from utilities.helpers import simple_Fourier_coeffs,trans_bearing_to_polar,find_relative_coords,rectangular_domain
# the North and East directions are impulses, so a large number of Fourier terms are required to represent them correctly
# =====================================
no_bins = 360 #72*400# 4*30000 seems to be the max
# ====================================

U_WB_i1, P_WB_i1,theta_WB_i1 = new_wr1(no_bins)
U_i1,P_i1,theta_i1 = trans_bearing_to_polar(U_WB_i1,P_WB_i1,theta_WB_i1)

#U_i,P_i,theta_i,_ = get_floris_wind_rose(6)
num_aep = JF_num(U_i1,P_i1,theta_i1,layout,turb,K)

Fourier_coeffs1_Ct0,_ = simple_Fourier_coeffs(U_i1*P_i1*len(P_i1)/(2*np.pi)) 
c_0,_,_ = Fourier_coeffs1_Ct0 

Fourier_coeffs1_Ct,_ = simple_Fourier_coeffs((1 - np.sqrt(1 - turb.Ct_f(U_i1))) * U_i1*P_i1*len(P_i1)/(2*np.pi)) 
a_0,a_n,b_n = Fourier_coeffs1_Ct
xx,yy,plot_points,_,_ = rectangular_domain(layout)
flower_aep = np.sum(flowers_2(Fourier_coeffs1_Ct,layout,layout,turb,K,c_0))

from utilities.flowers_interface import FlowersInterface
flower_int = FlowersInterface(U_WB_i1,P_WB_i1,np.rad2deg(theta_WB_i1), layout, turb,num_terms=no_bins//2, k=K) 
LC_flower_aep = np.sum(flower_int.calculate_aep())

print("=== Test 5A ===")
print(f"simple_aep       : {simple_aep:.6f}")
print(f"numerical aep    : {num_aep:.6f}")
print(f"my Jensen FLOWERS: {flower_aep:.6f} (with {len(U_i1)} bins) ({pce(num_aep,flower_aep):+.3f})%")

#%% they should also agree for a more complex wind rose ...
from utilities.helpers import get_floris_wind_rose_WB,rectangular_layout,pce
layout = rectangular_layout(4,7,np.deg2rad(20))
U_WB_i4,P_WB_i4,theta_WB_i4,_ = get_floris_wind_rose_WB(4,wd=np.arange(0, 360, 1.0))
U_i4,P_i4,theta_i4 = trans_bearing_to_polar(U_WB_i4,P_WB_i4,theta_WB_i4)

num_aep2 = JF_num(U_i4,P_i4,theta_i4,layout,turb,K)

Fourier_coeffs1_Ct,_ = simple_Fourier_coeffs((1 - np.sqrt(1 - turb.Ct_f(U_i4))) * U_i4*P_i4*len(P_i4)/(2*np.pi)) 
a_0,a_n,b_n = Fourier_coeffs1_Ct
c_0 = np.sum(U_i4*P_i4)/np.pi
flower_aep2 = np.sum(flowers_2(Fourier_coeffs1_Ct,layout,layout,turb,K,c_0))

flower_int = FlowersInterface(U_WB_i4,P_WB_i4,np.rad2deg(theta_WB_i4), layout, turb,num_terms=len(a_n), k=K) 
LC_flower_aep2 = np.sum(flower_int.calculate_aep())

print("=== Test 5B ===")
print(f"numerical aep    : {num_aep2:.6f}")
print(f"my Jensen FLOWERS: {flower_aep2:.6f} (with {len(U_i4)} bins) ({pce(np.sum(num_aep2),np.sum(flower_aep2)):+.3f})%")

#%% wake "shape" validation

def Uw(U_i,du):
    return U_i*(1-du)

U_i = 10
import matplotlib.pyplot as plt

from matplotlib import cm 
x = np.linspace(-20,20,1000)
y = K*x+0.5 #cartesian wake boundary
xx, yy = np.meshgrid(np.linspace(-20,20,1000),np.linspace(-20,20,1000))
R = np.sqrt(xx**2+yy**2)
THETA = np.arctan2(yy,xx)
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
du = U_delta_J3(U_i,R,THETA,K)
du = np.where(np.sqrt(xx**2+yy**2)>1,du,0)
cf = ax.contourf(xx,yy,Uw(U_i,du),50,cmap=cm.coolwarm)
fig.colorbar(cf)
ax.plot(x,y,color='black')