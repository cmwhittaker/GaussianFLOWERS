#SANITY CHECK POWER GENERATION
#Consider the farm layout:
#
#       x
# 
# 
#
#     x   x
#
# (k=0.03)
# coords: (0,0) (-0.2,-3) (+0.2,-3)
# subject to a 1)Northern and 2)Easterly wind with U_inf = 13 
# with equal probabilities (P_1 = P_2 = 0.5)
# P_a,P_b,P_c,P_b is power generation from North,East,South,West
#
# The total power generation must be 0.5*P_a + 0.5*P_b
# But if I made a mistake in the theta coordinate system (e.g. theta = -theta), 
# this would give the same power generation.
#
# If the farm is rotated 90 deg clockwise, and the power generation will again 
# be 0.5*P_a + 0.5*P_b
# The only mistake in coordinate system would be theta = -theta + 180 (this would
# produce the same power). 
# Since in both cases the power generation is the same (note that 2. would be 0.5*P_a + 0.5*P_c if the first coordinate system was wrong), the coordinate system must be correct
# 
# The power calculations were performed 1) by hand 2) using simplified functions 3) 
# actual aep functions. All methods agree, so I consider the functions validated.


#%% another simple sanity check
%load_ext autoreload
%autoreload 2

U_inf = 13

import numpy as np
from turbines_v01 import iea_10MW
turb = iea_10MW()
Ct_f = turb.Ct_f
Cp_f = turb.Cp_f

K=0.03 #this is user defined
ct = Ct_f(U_inf) #Ct for each direction
print("ct: {}".format(ct))
ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))
lim = (np.sqrt(ct/8)-ep)/K # the x limit
if lim < 0:
    lim = 0.01
print("lim: {}".format(lim))
x = 3 #downstream distance
y = 0.2

def U_delta(x,y):
      U_delta = (1-np.sqrt(1-(ct/(8*(K*x+ep)**2))))*(np.exp(-(y)**2/(2*(K*x+ep)**2)))
      return U_delta

def Pwr(U):
     alpha = ((0.5*1.225*9801*np.pi)/(1*10**6))
     return alpha*turb.Cp_f(U)*U**3

P_a = Pwr(U_inf)+2*Pwr(U_inf*(1-U_delta(3,0.2)))
P_b = 2*Pwr(U_inf) +Pwr(U_inf*(1-U_delta(0.4,0)))
P_c = 2*Pwr(U_inf) +Pwr(U_inf*(1-2*U_delta(3,0.2)))

#%% power using the numerical function
layout1 = np.array(((0,0),(-0.2,-3),(+0.2,-3)))
layout2 = np.array(((0,0),(-3,-0.2),(-3,+0.2)))
layout = layout1

U_i = np.array((13,13,))
P_i = np.array((0.5,0.5))
theta_i = np.array((0,np.pi/2))

aep2 = 0.5*P_a + 0.5*P_b

from AEP3_3_functions import num_F_v02
aep3,_,_ = num_F_v02(U_i,P_i,theta_i,layout,layout,turb,K=K,Ct_op = 1,Cp_op = 1)
print("aep2: {}".format(aep2))
print("aep3: {}".format(np.sum(aep3)))

#so, in theory, that validates the num_F_v02 method

#%% now use the num_F_v02 to validate ntag_PA_v03
def new_wr1(NO_BINS):
        if not NO_BINS%4 == 0:
              raise ValueError("Must be neatly divisible by 4") 
        theta_i = np.deg2rad(np.linspace(0,360,NO_BINS,endpoint=False))
        U_i = U_inf*np.ones(NO_BINS)
        P_i = np.zeros(NO_BINS)
        P_i[0], P_i[NO_BINS//4] = 0.5, 0.5
        return theta_i,U_i, P_i

theta_i,U_i, P_i = new_wr1(10*32)
#this should converge as the number of bins is increased!
#(as the Fourier series approaches two dirac delta functions...)
WAV_CT = np.sum(turb.Ct_f(U_i)*P_i)
from AEP3_3_functions import ntag_PA_v03,simple_Fourier_coeffs_v01
aep4,_,_ = num_F_v02(U_i,P_i,theta_i,layout,layout,turb,K=K,Ct_op=3,WAV_CT=WAV_CT,Cp_op=2,cross_ts=False,ex=False)
print("aep4: {}".format(np.sum(aep4)))
#numerical with Ct_op=3,Cp_op=2,cross_ts=False,ex=False is the discrete convolution equivalent of ntag
_,cjd3_PA_terms = simple_Fourier_coeffs_v01(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))

aep5,_ = ntag_PA_v03(cjd3_PA_terms,layout,layout,turb,WAV_CT,K)
print("aep5: {}".format(np.sum(aep5)))
#it does! so that's (in theory) also validated!

#%% some more testing to validate the old (cube the average) way of doing things
#start with a single, simple turbine ...
layout = np.array(((0,0),(0,5),(5,0),(5,5)))

def new_wr1(NO_BINS):
        if not NO_BINS%4 == 0:
              raise ValueError("Must be neatly divisible by 4") 
        theta_i = np.deg2rad(np.linspace(0,360,NO_BINS,endpoint=False))
        U_i = U_inf*np.ones(NO_BINS)
        P_i = np.zeros(NO_BINS)
        P_i[0], P_i[NO_BINS//4] = 0.5, 0.5
        return theta_i,U_i, P_i

theta_i,U_i, P_i = new_wr1(32*32)

_,cjd_noCp_PA_terms = simple_Fourier_coeffs_v01((P_i*U_i*len(P_i))/(2*np.pi))
from AEP3_3_functions import caag_PA_v03
aep6,_ = caag_PA_v03(cjd_noCp_PA_terms,layout,layout,turb,WAV_CT,K)
print("aep6: {}".format(np.sum(aep6)))
#this should be the same as:
WAV_CT = np.sum(turb.Ct_f(U_i)*P_i)
aep7,_,_ = num_F_v02(U_i,P_i,theta_i,layout,layout,turb,K=K,Ct_op=3,WAV_CT=WAV_CT,Cp_op=4,cross_ts=False,ex=False)
print("aep7: {}".format(np.sum(aep7)))

#%% testing the new cubeAv_v5 against num_F_v02
#layout = np.array(((0,0),(0,5),(5,0),(5,5)))
from AEP3_3_functions import cubeAv_v5,gen_local_grid_v01C
WAV_CT = np.sum(turb.Ct_f(U_i)*P_i)
#this should be the same as:
aep8,_,_ = num_F_v02(U_i,P_i,theta_i,layout,layout,turb,K=K,Ct_op=3,WAV_CT=WAV_CT,Cp_op=1,cross_ts=True,ex=True)
print("aep8: {}".format(np.sum(aep8)))

aep10,_ = cubeAv_v5(U_i,P_i,theta_i,
              layout,
              layout, 
              turb,
              RHO=1.225,K=K,
              u_lim=None,ex=True,Ct_op=3,WAV_CT=WAV_CT)

print("aep10: {}".format(np.sum(aep10)))