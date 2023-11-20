#%% 
# 7 shaped layout with num_F + vect_num_F
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

K=0.03 #wake expansion rate
alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst

U_inf1 = 15
U_inf2 = 13
U_i = np.array((U_inf1,U_inf2,))
P_i = np.array((0.7,0.3,))
theta_i = np.array((0,np.pi/2,))
from utilities.helpers import get_WAV_pp
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)
wav_ep = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))

layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))
#cnst thrust coeff (Ct_op=3),global power coeff (Cp_op=2), exclude cross terms (cross_ts=False), approx wake deficit (ex=False)
from utilities.AEP3_functions import num_Fs,vect_num_F
pow_j1,_,_ = num_Fs(U_i,P_i,theta_i,
                    layout,layout,
                    turb,
                    K,
                    Ct_op=2,
                    Cp_op=1,
                    cross_ts=True,cube_term=True,ex=True)
 
pow_j2,_ = vect_num_F(U_i,P_i,theta_i,
                      layout,layout, 
                      turb,
                      K,
                      Ct_op=2,
                      Cp_op=1,  
                      ex=True)

print("=== Test 4A ===")
print(f"num_F      aep: {np.sum(pow_j1):.6f}")
print(f"vect_num_F aep: {np.sum(pow_j2):.6f}")


#%% the two functions should also agree for a more complex wind rose (layout is unchanged)
from utilities.helpers import get_floris_wind_rose,pce,simple_Fourier_coeffs
site_no = 2 #vary between 1-12 (inclusive)
U_i,P_i = get_floris_wind_rose(site_no,wd=np.arange(0, 360, 1))
theta_i = np.linspace(0,2*np.pi,len(U_i))
_,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/((2*np.pi)))
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)

pow_j1,_,_ = num_Fs(U_i,P_i,theta_i,
                    layout,layout,
                    turb,
                    K,
                    Ct_op=2,
                    Cp_op=1,
                    cross_ts=True,cube_term=True,ex=True)

pow_j2,_ = vect_num_F(U_i,P_i,theta_i,
                      layout,layout, 
                      turb,
                      K,
                      Ct_op=2,
                      Cp_op=1,  
                      ex=True)

print("=== Test 4B ===")
print("These should agree exactly")
print(f"num_F aep: {np.sum(pow_j1):.6f}")
print(f"ntag  aep: {np.sum(pow_j2):.6f} (with {len(U_i)} bins) ({pce(np.sum(pow_j1),np.sum(pow_j2)):+.3f})%")

