#%% 
# 2 turbine, uniform wind rose with ntag + num_F
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
no_bins = 4*72
U_inf1 = 15
P_1 = 1/(no_bins)
U_i = np.full(no_bins,U_inf1)  #each with same strength
P_i = np.full(no_bins,P_1)  #each equally likely
theta_i = np.linspace(0,2*np.pi,no_bins,endpoint=False)
from utilities.helpers import get_WAV_pp
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)
wav_ep = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))

#first, start with a simple implementation 





          # some simple calcs here




#Next check num_Fs is giving the same result
layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))
#cnst thrust coeff (Ct_op=3),global power coeff (Cp_op=2), exclude cross terms (cross_ts=False), approx wake deficit (ex=False)
from utilities.AEP3_functions import num_Fs,ntag_PA
pow_j1,_,_ = num_Fs(U_i,P_i,theta_i,
                   layout,layout,
                   turb,
                   K,
                   Ct_op=3,wav_Ct=wav_Ct,
                   Cp_op=2,
                   cross_ts=False,cube_term=False,ex=False)

from utilities.helpers import simple_Fourier_coeffs
_,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/((2*np.pi)))

pow_j2,_ = ntag_PA(cjd3_PA_terms,
            layout,layout,
            turb,
            K,
            wav_Ct,
            u_lim=1.9)

print("=== Test 3A ===")
print(f"num_F aep: {np.sum(pow_j1):.6f}")
print(f"ntag  aep: {np.sum(pow_j2):.6f} (with {no_bins} bins)")
