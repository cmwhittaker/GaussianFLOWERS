#%% validating ntag_PA

# check the wake "flow field" using contourf
# (theta is theta' NOT a wind bearing )
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
K=0.03 #wake expansion rate

def new_wr1(no_bins):
    # the FLOWERS method needs a large number of bins so that the spikes in the Fourier series are "sufficiently" narrow 
    if not no_bins%4 == 0:
        raise ValueError("no_bins must be a multiple of 4") 
    U_i1 = np.zeros(no_bins)
    P_i1 = np.zeros(no_bins)

    # U_i1[0] = 10 #WESTERLY wind direction
    # P_i1[0] = 1 #one direction

    U_i1[no_bins//16] = 25 #22.5 anticlock shift
    P_i1[no_bins//16] = 1 #still one direction

    # two-direction test
    # U_i1[0],U_i1[(no_bins)//4] = 15,15
    # P_i1[0],P_i1[(no_bins)//4] = 0.5, 0.5 

    theta_i1 = np.linspace(0,2*np.pi,no_bins,endpoint=False)
    return U_i1, P_i1, theta_i1

U_i,P_i,theta_i = new_wr1(360)

from utilities.AEP3_functions import num_Fs,ntag_PA
from utilities.helpers import rectangular_domain,simple_Fourier_coeffs,get_WAV_pp,find_relative_coords

layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))
xx,yy,plot_points,_,_ = rectangular_domain(layout,xr=200,yr=200)

_,Fourier_coeffs3_PA = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)

_,ff = ntag_PA(Fourier_coeffs3_PA,
                   plot_points,layout,
                   turb,
                   K,
                   wav_Ct,
                   u_lim=0.5,
                   flo_vis=True)

#visualise / plot
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')

cf = ax.contourf(xx,yy,ff.reshape(xx.shape),20,cmap=cm.coolwarm)
ax.scatter(layout[:,0],layout[:,1],marker='x',color='black')
fig.colorbar(cf)

# 
# using num_F to validate ntag ( also against "manual" calculations )
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

K=0.03 #wake expansion rate
alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst

U_inf1 = 15
U_inf2 = 13
U_WB_i = np.array((U_inf1,U_inf2,))
P_WB_i = np.array((0.7,0.3,))
theta_WB_i = np.array((0,np.pi/2,))
from utilities.helpers import get_WAV_pp
wav_Ct = get_WAV_pp(U_WB_i,P_WB_i,turb,turb.Ct_f)
wav_ep = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))

#first, start with a simple implementation 
def U_delta_SA(o,a): 
    #"dumb" way to convert to polar
    #a:adjacent to angle, o:opposite to angle
    r = np.sqrt(a**2+o**2)
    theta = np.arctan2(o,a)
    #wake velocity deficit with small angle
    #use weight-averaged globals for Ct and ep
    U_delta = (1-np.sqrt(1-(wav_Ct/(8*(K*r*1+wav_ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+wav_ep)**2)))   
    U_delta = np.where(r<2,0,U_delta) #stop self-produced
    return U_delta

def Pwr_NC(U_inf,delta): 
     #ns: number of turbine in superposistion
     #power neglecting the cross terms (and without cubic term)
     #and with a global power coefficient
     U_cube = U_inf**3*(1-3*np.sum(delta)+3*np.sum(delta**2))
     return alpha*turb.Cp_f(U_inf)*U_cube

# === North @ U_inf1 ===
#T0, T1 are unwaked
DU_nT0T1 = 0
#T2 is waked by T1
DU_nT2 = U_delta_SA(0.2,3)
#T3 is waked by T1 and T2
DU_nT3 = np.array((U_delta_SA(0.4,6),U_delta_SA(0.2,3)))
#Total power is found from the wake velocities
P_n =  2*Pwr_NC(U_WB_i[0],DU_nT0T1) + Pwr_NC(U_WB_i[0],DU_nT2) + Pwr_NC(U_WB_i[0],DU_nT3)

# === East @ U_inf2 ===
#T1,T2,T3 are unwaked
DU_eT1T2T3 = 0
#T0 is waked by T1
DU_eT0 = U_delta_SA(0,3)
#Total power is found from the wake velocities
P_e =  3*Pwr_NC(U_WB_i[1],DU_eT1T2T3)+Pwr_NC(U_WB_i[1],DU_eT0)

# === Total power ===
simple_aep = P_WB_i[0]*P_n+P_WB_i[1]*P_e

#Next check num_Fs is giving the same result
layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))
#cnst thrust coeff (Ct_op=3),global power coeff (Cp_op=2), exclude cross terms (cross_ts=False), approx wake deficit (ex=False)
from utilities.helpers import trans_bearing_to_polar
U_i,P_i,theta_i = trans_bearing_to_polar(U_WB_i,P_WB_i,theta_WB_i)

pow_j1,_,_ = num_Fs(U_i,P_i,theta_i,
                   layout,layout,
                   turb,
                   K,
                   Ct_op=3,wav_Ct=wav_Ct,
                   Cp_op=2,
                   cross_ts=False,cube_term=False,ex=False)

#Next check ntag gives the same result
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

no_bins = 4*360
U_WB_i2, P_WB_i2,theta_WB_i2 = new_wr1(no_bins)
U_i2,P_i2,theta_i2 = trans_bearing_to_polar(U_WB_i2,P_WB_i2,theta_WB_i2)

_,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i2)*(P_i2*(U_i2**3)*len(P_i2))/((2*np.pi)))

pow_j2,_ = ntag_PA(cjd3_PA_terms,
            layout,layout,
            turb,
            K,
            wav_Ct,
            u_lim=1.9)

print("=== Test 2A ===")
print("These should agree to ~6sf")
print(f"simple aep: {simple_aep:.6f}")
print(f"num_F  aep: {np.sum(pow_j1):.6f}")
print(f"ntag   aep: {np.sum(pow_j2):.6f} (with {no_bins} bins)")

#%% the two functions should also agree for a more complex wind rose (layout is unchanged)
from utilities.helpers import get_floris_wind_rose,pce,rectangular_layout
layout = rectangular_layout(10,7,np.deg2rad(10))
import warnings
result = []
print("Testing with all 12 wind roses, this should take less than 30s")
for i in range(12):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=FutureWarning)

        U_i,P_i,theta_i,_ = get_floris_wind_rose(i+1,wd=np.arange(0, 360, 1))

        _,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/((2*np.pi)))
        wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)
        pow_j1,_,_ = num_Fs(U_i,P_i,theta_i,
                        layout,layout,
                        turb,
                        K,
                        Ct_op=3,wav_Ct=wav_Ct,
                        Cp_op=2,
                        cross_ts=False,cube_term=False,ex=False)

        pow_j2,_ = ntag_PA(cjd3_PA_terms,
                    layout,layout,
                    turb,
                    K,
                    wav_Ct,
                    u_lim=1.9)
        result.append(pce(np.sum(pow_j1),np.sum(pow_j2)))

print("=== Test 2B ===")
print("Comparing num_F with ntag. The difference (pce) should be negligible:")
print(f"{np.mean(result):+.3f}% mean pce error across 12 sites (with {len(U_i)} bins) ")