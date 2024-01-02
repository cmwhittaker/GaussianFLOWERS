#%% Test 5
# 
# using a numerical equivalent of Jensen FLOWERS (JF_num) to validate the Jensen FLOWERS method ( also against "manual" calculations )
#
# U_i = [15,13]
# P_i = [0.7,0.3]
# theta_i = [0,np.pi/2] (wind bearings)
# 7 shaped layout 
# layout: np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))
# looks like:
# x        x
#          
#         x
# 
#        x

import sys
import os
sys.path.append(os.path.join('..', 'src')) #allow to import f2rom utilities (there may be a better way ...)

if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2
import numpy as np
from utilities.turbines import iea_10MW
turb = iea_10MW()

K=0.05 #wake expansion rate

def U_delta_J(U_i,x,y,K):
    # U delta Jesen (cartesian)
    Ct = turb.Ct_f(U_i) #base thrust coefficent on free stream
    du = ((1-np.sqrt(1-Ct))/(2*K*x+1)**2) # 
    w = np.where((np.abs(y)<K*x+1/2)&(x>=0),1,0)  
    return du*w

def U_delta_J2(U_i,x,y,K,r_lim=0.5):
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
    w = np.where(r<r_lim,0,w) #no wake in rotor diameter (stop self-produced wakes)
    return du*w

def U_delta_J3(U_i,r,theta,K,r_lim=0.5):
    # U delta Jesen using 2nd Order Taylor approximation
    # using POLAR coordinates
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
    #second order taylor series approximation of the Jesen wake model
    Ct = turb.Ct_f(U_i) #base thrust coefficent on free stream
    du = (1-np.sqrt(1-Ct))*(1/(2*K*r+1)**2+(2*K*r*theta**2)/(2*K*r+1)**3)
    theta_c = np.arctan2((1 / (2*r) + K * np.sqrt(1 + K**2 - (2*r)**(-2))),(-K / (2*r) + np.sqrt(1 + K**2 - (2*r)**(-2)))) 
    w = np.where((-theta_c<=theta)&(theta<=theta_c),1,0)
    w = np.where(r<r_lim,0,w) #no wake in rotor diameter (stop self-produced wakes)
    return du*w

#check the "shape" of the wakes
import matplotlib.pyplot as plt
# Creating a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3),dpi=400)

from matplotlib import cm 
x = np.linspace(-20,20,1000)
y = K*x+0.5 #cartesian wake boundary
xx, yy = np.meshgrid(np.linspace(-20,20,1000),np.linspace(-20,20,1000))
R = np.sqrt(xx**2+yy**2)
THETA = np.arctan2(yy,xx)
rtr_mask = np.where(np.sqrt(xx**2+yy**2)<0.5,1,0)

du1 = np.where(rtr_mask,0,U_delta_J(15,xx,yy,K))
ax1.contourf(xx,yy,15-du1,50,cmap=cm.coolwarm)
ax1.plot(x,y)

du2 = np.where(rtr_mask,0,U_delta_J2(15,xx,yy,K))
ax2.contourf(xx,yy,15-du2,50,cmap=cm.coolwarm)
ax2.plot(x,y)

du3 = np.where(rtr_mask,0,U_delta_J3(15,R,THETA,K))
ax3.contourf(xx,yy,15-du3,50,cmap=cm.coolwarm)
ax3.plot(x,y)

#%% next is numerical implementation of JENSEN FLOWERS

from utilities.helpers import trans_bearing_to_polar,rectangular_domain,find_relative_coords,get_floris_wind_rose,get_floris_wind_rose_WB
def JF_num(U_i,P_i,theta_i,plot_points,layout,turb,K,RHO=1.225):
    #numerical convolution using the 2nd order taylor approximation of the Jesen wake model
    r_jk,theta_jk = find_relative_coords(plot_points,layout)

    U_wav = np.sum(U_i*P_i)
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None]
    r_ijk =  np.broadcast_to(r_jk[None,:,:],theta_ijk.shape)
    U_ijk = np.broadcast_to(U_i[:,None,None],theta_ijk.shape)
    P_ijk = np.broadcast_to(P_i[:,None,None],theta_ijk.shape)

    DU_wav = np.sum(U_ijk*P_ijk*(U_delta_J3(U_ijk,r_ijk,theta_ijk,K,r_lim=0.5)),axis=(0,2))

    U_w_wav = U_wav - DU_wav
    alpha = ((0.5*RHO*turb.A)/(1*10**6)) #turbine cnst
    aep = alpha*turb.Cp_f(U_w_wav)*U_w_wav**3
    return aep,U_w_wav

layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))

# check the wake "flow field" using contourf
# (theta is theta' NOT a wind bearing )

# U_i,P_i,theta_i = np.array((4,)),np.array((1,)),np.array((np.deg2rad(0),)) #single Westerly with 10 deg
U_i,P_i,theta_i,_ = get_floris_wind_rose(6)

# U_i,P_i,theta_i = np.array((U_inf1,U_inf1)),np.array((0.5,0.5)),np.array((np.deg2rad(0),np.deg2rad(90))) #West/East at same speed
xx,yy,plot_points,_,_ = rectangular_domain(layout,xr=200,yr=200)
pow_j,ff = JF_num(U_i,P_i,theta_i,plot_points,layout,turb,K,RHO=1.225)
#visualise / plot
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')

cf = ax.contourf(xx,yy,ff.reshape(xx.shape),20,cmap=cm.coolwarm)
ax.scatter(layout[:,0],layout[:,1],marker='x',color='black')
fig.colorbar(cf)

#%% next is FLOWERS 

import warnings
# Suppress all runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utilities.helpers import simple_Fourier_coeffs,trans_bearing_to_polar,rectangular_domain,pce
# =====================================
no_bins = 360 #72*400# 4*30000 seems to be the max
# ====================================

from utilities.AEP3_functions import jflowers

def new_wr1(no_bins):
    # the FLOWERS method needs a large number of bins so that the spikes in the Fourier series are "sufficiently" narrow 
    if not no_bins%4 == 0:
        raise ValueError("no_bins must be a multiple of 4") 
    U_i1 = np.zeros(no_bins)
    P_i1 = np.zeros(no_bins)
    U_i1[0] = 4 #WESTERLY wind direction
    P_i1[0] = 1 #one direction

    # two-direction test
    # U_i1[0],U_i1[(no_bins)//4] = 15,15
    # P_i1[0],P_i1[(no_bins)//4] = 0.5, 0.5 

    theta_i1 = np.linspace(0,2*np.pi,no_bins,endpoint=False)
    return U_i1, P_i1, theta_i1


U_i1,P_i1,theta_i1 = new_wr1(no_bins)

c_0 = np.sum(U_i1*P_i1)/np.pi
Fourier_coeffs1_Ct,_ = simple_Fourier_coeffs((1 - np.sqrt(1 - turb.Ct_f(U_i1))) * U_i1*P_i1*len(P_i1)/(2*np.pi)) 

pow_j2,ff2 = jflowers(Fourier_coeffs1_Ct,plot_points,layout,turb,K,c_0)

#visualise / plot
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')
ax.scatter(layout[:,0],layout[:,1],marker='x')
cf = ax.contourf(xx,yy,ff2.reshape(xx.shape),20,cmap=cm.coolwarm)
ax.scatter(layout[:,0],layout[:,1],marker='x',color='black')
fig.colorbar(cf)

#%% next is "manually" calculated aep vs numerical aep vs flowers_aep
# for a "simple wind rose and layout"

layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))

U_inf1 = 15
U_inf2 = 14

U_WB_i = np.array((U_inf1,U_inf2,))
P_WB_i = np.array((0.7,0.3,))
theta_WB_i = np.array((0,np.pi/2,)) #wind bearing thetas

# MANUAL CALCULATIONS FIRST
# === North @ U_inf1 ===
#T0, T1 are unwaked
DU_nT0T1 = 0
#T2 is waked by T1
DU_nT2 = U_delta_J2(U_WB_i[0],3,0.2,K)
#T3 is waked by T1 and T2
DU_nT3 = U_delta_J2(U_WB_i[0],6,0.4,K)+U_delta_J2(U_WB_i[0],3,0.2,K)
#wa velocity deficit for the North direction 
DU_wav_n = U_WB_i[0]*P_WB_i[0]*np.array((DU_nT0T1,DU_nT0T1,DU_nT2,DU_nT3))

# === East @ U_inf2 ===
#T1,T2,T3 are unwaked
DU_eT1T2T3 = 0
#T0 is waked by T1
DU_eT0 = U_delta_J2(U_WB_i[1],3,0,K)
#wa velocity deficit for the East direction 
DU_wav_e = U_WB_i[1]*P_WB_i[1]*np.array((DU_eT0,DU_eT1T2T3,DU_eT1T2T3,DU_eT1T2T3))

# === Totals ===
#weight-averaged total deficit
Ui_wav = np.sum(U_WB_i*P_WB_i) #weight average velocity
Uw_wav_t = Ui_wav - (DU_wav_n+DU_wav_e)
#power from weight-averaged velocities
alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst
smp_aep1 = np.sum(alpha*turb.Cp_f(Uw_wav_t)*Uw_wav_t**3)

#numerical next
#convert wind bearing theta into polar coordinate theta
U_i,P_i,theta_i = trans_bearing_to_polar(U_WB_i,P_WB_i,theta_WB_i)
num_aep1,_ = JF_num(U_i,P_i,theta_i,layout,layout,turb,K,RHO=1.225)

#FLOWERS next
#flowers requires discrete bins across the entire range 0 to 2pi
def new_wr2(no_bins):
    # the FLOWERS method needs a large number of bins so that the spikes in the Fourier series are "sufficiently" narrow 
    #(theta here in wind bearing coordinates)
    if not no_bins%4 == 0:
        raise ValueError("no_bins must be a multiple of 4") 
    U_WB_i2 = np.zeros(no_bins)
    P_WB_i2 = np.zeros(no_bins)
    U_WB_i2[0],U_WB_i2[no_bins//4] = U_WB_i[0], U_WB_i[1]
    P_WB_i2[0],P_WB_i2[no_bins//4] = P_WB_i[0], P_WB_i[1]

    theta_WB_i2 = np.linspace(0,2*np.pi,no_bins,endpoint=False)
    return U_WB_i2, P_WB_i2, theta_WB_i2

# =====================================
no_bins = 100*360# 4*30000 seems to be the max
# ====================================

#create equivalent wind rose
U_WB_i2, P_WB_i2, theta_WB_i2 = new_wr2(no_bins)
U_i2,P_i2,theta_i2 = trans_bearing_to_polar(U_WB_i2, P_WB_i2, theta_WB_i2)

c_02 = np.sum(U_i2*P_i2)/np.pi
Fourier_coeffs2_Ct,_ = simple_Fourier_coeffs((1 - np.sqrt(1 - turb.Ct_f(U_i2))) * U_i2*P_i2*len(P_i2)/(2*np.pi)) 

flr_aep1,_ = jflowers(Fourier_coeffs2_Ct,layout,layout,turb,K,c_02)

print("=== Simple AEP test ===")
print("The Jensen FLOWERS result should converge as the number of bins is increased")
print(f"simple_aep       : {smp_aep1:.6f}")
print(f"numerical aep    : {np.sum(num_aep1):.6f}")
print(f"my Jensen FLOWERS: {np.sum(flr_aep1):.6f} (with {len(U_WB_i2)} bins) ({pce(np.sum(num_aep1),np.sum(flr_aep1)):+.3f}%)")

#also using Locasio's version(different theta coord system)
from utilities.flowers_interface import FlowersInterface
flower_int = FlowersInterface(U_i2,P_i2,np.rad2deg(3*np.pi/2-theta_i2), layout, turb,num_terms=no_bins//2, k=K) 
LC_flower_aep = np.sum(flower_int.calculate_aep())
print(f"LC               : {LC_flower_aep:.6f}") 

#%% next, although it is no longer possible to do a "manual" calculation, the methods should agree for a more complex (realistic) wind rose and layout

from utilities.helpers import get_floris_wind_rose,rectangular_layout,pce
# error is quite sensitive to the spacing
layout = rectangular_layout(10,7,np.deg2rad(10))

site_no = 7
U_i4,P_i4,theta_i4,_ = get_floris_wind_rose(site_no,wd=np.arange(0, 360, 1.0))

num_aep4,_ = JF_num(U_i4,P_i4,theta_i4,layout,layout,turb,K)

Fourier_coeffs4_Ct,_ = simple_Fourier_coeffs((1 - np.sqrt(1 - turb.Ct_f(U_i4))) * U_i4*P_i4*len(P_i4)/(2*np.pi)) 
a_0,a_n,b_n = Fourier_coeffs4_Ct
c_04 = np.sum(U_i4*P_i4)/np.pi
flr_aep4,_ = jflowers(Fourier_coeffs4_Ct,layout,layout,turb,K,c_04)

from utilities.flowers_interface import FlowersInterface
#also using Locasio's version(different theta coord system)
flower_int = FlowersInterface(U_i4,P_i4,np.rad2deg(3*np.pi/2-theta_i4), layout, turb,num_terms=len(a_n), k=K) 
LC_flower_aep2 = np.sum(flower_int.calculate_aep())

print("=== Realistic/Complex AEP test ===")
print(f"numerical aep    : {np.sum(num_aep4):.6f}")
print(f"my Jensen FLOWERS: {np.sum(flr_aep4):.6f} (with {len(U_i4)} bins) ({pce(np.sum(num_aep4),np.sum(flr_aep4)):+.3f}%)")
print(f"LC2              : {LC_flower_aep2:.6f}")

#%%

layout = rectangular_layout(6,7,np.deg2rad(0))

def rotate_layout_L(layout,rot):
    #rotates layout anticlockwise by angle rot 
    Xt,Yt = layout[:,0],layout[:,1]
    rot_Xt = Xt * np.cos(rot) - Yt * np.sin(rot)
    rot_Yt = Xt * np.sin(rot) + Yt * np.cos(rot) 
    layout_r = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
    return layout_r

layout_r = rotate_layout_L(layout,np.deg2rad(0))
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')
ax.scatter(layout_r[:,0],layout_r[:,1])

#%% just checking if the Fourier series agree

xs = np.linspace(0,2*np.pi,360,endpoint=False)
ys = (1 - np.sqrt(1 - turb.Ct_f(U_i4))) * U_i4*P_i4*len(P_i4)/(2*np.pi)
Fourier_coeffs4_Ct,_ = simple_Fourier_coeffs(ys) 
a_0,a_n,b_n = Fourier_coeffs4_Ct

t = np.arange(1,len(a_n)+1,1)
yr = a_0/2 + np.sum(a_n[None,:]*np.cos(t[None,:]*xs[:,None])+b_n[None,:]*np.sin(t[None,:]*xs[:,None]),axis=-1)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')
ax.scatter(xs,ys,marker='x')
ax.scatter(xs,yr,marker='+')