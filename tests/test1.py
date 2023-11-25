#%% 
# 7 shaped layout with num_F
# subject to
# U_i = [15,13]
# P_i = [0.7,0.3]
# theta_i = [0,np.pi/2] (wind bearings)

import sys
import os
sys.path.append(os.path.join('..', 'src')) #allow to import from utilities (there may be a better way ...)
import sys
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2

import numpy as np
from utilities.turbines import iea_10MW
turb = iea_10MW()

K=0.03 #wake expansion rate
alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst
U_i = np.array((15,13,))
P_i = np.array((0.7,0.3))
theta_WB_i = np.array((0,np.pi/2))
layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))

#simple implementation of the Gaussian wake model
def U_delta(x,y,U_inf):
    #local wake coefficient!
    ct = turb.Ct_f(U_inf)
    ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))
    lim = (np.sqrt(ct/8)-ep)/K # the x limit
    if lim < 0.01:
        lim = 0.01 #stop self produced wake
    U_delta = (1-np.sqrt(1-(ct/(8*(K*y+ep)**2))))*(np.exp(-x**2/(2*(K*y+ep)**2)))
    U_delta = np.where(y>lim,U_delta,0)
    return U_delta

def Pwr(U): #power given local wake velocity
    return alpha*turb.Cp_f(U)*U**3

# === North @ U_inf1 ===
#T0, T1 are unwaked
Uw_nT0T1 = U_i[0]
#T2 is waked by T1
Uw_nT2 = U_i[0]*(1-U_delta(0.2,3,Uw_nT0T1))
#T3 is waked by T1 and T2
Uw_nT3 = U_i[0]*(1-(U_delta(0.4,6,Uw_nT0T1)+U_delta(0.2,3,Uw_nT2)))
#Total power is found from the wake velocities
P_n =  2*Pwr(Uw_nT0T1)+Pwr(Uw_nT2)+Pwr(Uw_nT3)

# === East @ U_inf2 ===
#T1,T2,T3 are unwaked
Uw_eT0T12 = U_i[1]
#T0 is waked by T1
Uw_eT2 = U_i[1]*(1-U_delta(0,3,Uw_eT0T12))
#Total power is found from the wake velocities
P_e =  3*Pwr(Uw_eT0T12)+Pwr(Uw_eT2)

from utilities.helpers import trans_bearing_to_polar
U_i,P_i,theta_i = trans_bearing_to_polar(U_i,P_i,theta_WB_i)

#Next check num_Fs is giving the same result
#local thrust coeff (Ct_op=1),local power coeff (Cp_op=1), include cross terms (cross_ts=True), exact wake deficit (ex=True)
from utilities.AEP3_functions import num_Fs
pow_j,_,_ = num_Fs(U_i,P_i,theta_i,
                   layout,layout,
                   turb,
                   K,
                   Ct_op=1,
                   Cp_op=1,
                   cross_ts=True,ex=True)
print("=== Test 1 ===")
print(f"simple aep: {P_i[0]*P_n+P_i[1]*P_e:.6f}")
print(f"num_F  aep: {np.sum(pow_j):.6f}")


