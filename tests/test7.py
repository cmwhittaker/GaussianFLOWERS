#%% Testing if the truncating Foureier series is working correctly
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append(os.path.join('..', 'src')) #allow to import from utilities (there may be a better way ...)
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2
from utilities.turbines import iea_10MW
turb = iea_10MW()

from utilities.helpers import random_layouts,fixed_rectangular_domain
np.random.seed(1)
K=0.03

# wind speed is uniform
theta_i = np.linspace(0,2*np.pi,endpoint=False)
U_i = 4*np.ones_like(theta_i)

X,Y,plot_points = fixed_rectangular_domain(15,r=200)
layout = np.array([[0,0],[0,2.5]])

probablity_shape = 1 + np.cos(theta_i)
#probablity_shape = np.ones_like(theta_i)

P_i = probablity_shape/(np.sum(probablity_shape))

from utilities.helpers import get_WAV_pp
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)

from utilities.AEP3_functions import num_Fs,ntag_PA
pow_j1, b, Uwff_j = num_Fs(U_i,P_i,
                           theta_i,
                           plot_points,layout,
                           turb,K,
                           u_lim=None,
                           Ct_op=3, wav_Ct=wav_Ct, 
                           Cp_op=2, #local Cp
                           cross_ts=False,ex=False,cube_term=False,
                           ff=False)

from utilities.helpers import simple_Fourier_coeffs
_,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/((2*np.pi)))

pow_j2,_ = ntag_PA(cjd3_PA_terms,
            layout,layout,
            turb,
            K,
            wav_Ct,
            u_lim=1.9)

print(f"pow_j1:{np.sum(pow_j1):.5f}, pow_j2:{np.sum(pow_j1):.5f}")
