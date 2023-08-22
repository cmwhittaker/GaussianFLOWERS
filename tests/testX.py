#%% this is a work in progress script containing a number of test cells (ipy) I'm still finalising. I'm reasonably confident that what I have is correct but I have not yet completed comphrehensive testing. 

#these cells don't even work right now

#%% then the next test ...

import sys
import os
sys.path.append(os.path.join('..', 'src')) #allow to import from utilities (there may be a better way ...)

import numpy as np
from utilities.turbines import iea_10MW
turb = iea_10MW()

K=0.03 #wake expansion rate
U_i = np.array((15,13))
P_i = np.array((0.5,0.5))
theta_i = np.array((0,np.pi/2))
layout = np.array(((0,0),(-0.2,-3),(+0.2,-3)))
from utilities.helpers import get_WAV_pp
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)
wav_ep = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))

def U_delta2(x,y): #slightly modified
      ct = wav_Ct
      ep = wav_ep
      U_delta = (1-np.sqrt(1-(ct/(8*(K*y+ep)**2))))*(np.exp(-x**2/(2*(K*y+ep)**2)))
      U_delta = np.where(y>lim,U_delta,0)
      return U_delta
def Pwr2(U_inf,delta): #power given local wake velocity
     #now neglecting the cross terms (without cubic term)
     U_cube = U_inf**3*(1-delta+3*delta**2)
     return alpha*turb.Cp_f(U_inf)*U_cube
#Northerly 15ms: 1xundisturbed, 2xsingularly waked
P_n = Pwr2(U_i[0],0)+2*Pwr2(U_i[0],U_delta2(0.2,3))
#Westerly 13ms: 2xundisturbed, 1xsingularly waked
P_w = 2*Pwr2(U_i[1],0)+Pwr2(U_i[1],U_delta2(0,0.4))
#total power:
P_t2 = 0.5*P_n+0.5*P_w
print(f"hand power check 2 aep: {P_t2:.4f}")

#this should agree with num_F
pow_j2,_,_ = num_Fs(U_i,P_i,theta_i,
                   layout,layout,
                   turb,
                   K,
                   Ct_op=3,wav_Ct = wav_Ct, #cnst
                   Cp_op=2, #global
                   cross_ts=False,
                   cube_term=False,
                   ex=True)

print(f"num_F power check aep: {np.sum(pow_j2):.4f}")

#%% now check ntag using num_Fs

import sys
import os
sys.path.append(os.path.join('..', 'src')) #allow to import from utilities (there may be a better way ...)
from utilities.turbines import iea_10MW
turb = iea_10MW()
layout = np.array(((0,0),(-0.2,-3),(+0.2,-3)))
K=0.03

import numpy as np
U_inf = 13
def new_wr1(NO_BINS):
      if not NO_BINS%4 == 0:
            raise ValueError("Must be neatly divisible by 4") 
      theta_i = np.deg2rad(np.linspace(0,360,NO_BINS,endpoint=False))
      U_i = U_inf*np.ones(NO_BINS)
      P_i = np.zeros(NO_BINS)
      P_i[0], P_i[NO_BINS//4] = 0.5, 0.5
      return theta_i,U_i, P_i

#this should converge as the number of bins is increased!
#(as the Fourier series approaches two dirac delta functions...)
from utilities.helpers import simple_Fourier_coeffs,get_WAV_pp
from utilities.AEP3_functions import num_Fs,ntag_PA
multipliers = [1,2,10]
for m in multipliers:
      theta_i,U_i, P_i = new_wr1(m*36)
      wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)

      #numerical with Ct_op=3,Cp_op=2,cross_ts=False,ex=False,cube_term=False is the discrete convolution equivalent of ntag
      aep4,_,_ = num_Fs(U_i,P_i,theta_i,
                        layout,layout,
                        turb,
                        K,
                        Ct_op=3,wav_Ct=wav_Ct,
                        Cp_op=2,
                        cross_ts=False,ex=False,cube_term=False)

      
      _,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))
      aep5,_ = ntag_PA(cjd3_PA_terms,
                 layout,layout,
                 turb,
                 K,
                 wav_Ct)

      print(f'with {str(m*36)} bins:')
      print("num_f aep4: {}".format(np.sum(aep4)))
      print("ntag  aep5: {}".format(np.sum(aep5)))

print("the two *should* converge ")
#%%and for a reasonably fine, realistc, wind rose (72 bins), the results should be close
from utilities.helpers import get_floris_wind_rose
U_i,P_i = get_floris_wind_rose(6)
theta_i = np.linspace(0,360,72,endpoint=False)

a1,_,_ = num_Fs(U_i,P_i,np.deg2rad(theta_i),
                layout,layout,
                turb,
                K,
                Ct_op=3,wav_Ct=wav_Ct,
                Cp_op=2,
                cross_ts=False,ex=False,cube_term=False)
_,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))
a2,_ = ntag_PA(cjd3_PA_terms,
               layout,layout,
               turb,
               K,
               wav_Ct)
print("a1: {}".format(np.sum(a1)))
print("a2: {}".format(np.sum(a2)))
#they are reasonably close
#%% num_Fs and vect_num_F should agree 
from utilities.AEP3_functions import vect_num_F
aep6,_,_ = num_Fs(U_i,P_i,theta_i,
                  layout,layout,
                  turb,
                  K,
                  Ct_op=2,wav_Ct=None,
                  Cp_op=1,
                  cross_ts=True,ex=True,cube_term=True)

aep7,_ = vect_num_F(U_i,P_i,theta_i,
                      layout,layout, 
                      turb,
                      K,
                      Ct_op=2,wav_Ct=None,
                      Cp_op=1,  
                      ex=True)

print("aep6: {}".format(np.sum(aep6)))
print("aep7: {}".format(np.sum(aep7)))

#%% num_Fs and caag should converge (as the Fourier series converges to two dirac delta functions) 
from utilities.AEP3_functions import caag_PA
multipliers = [1,2,10]
for m in multipliers:
      theta_i,U_i, P_i = new_wr1(m*16)
      wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)

      #numerical with Ct_op=3,Cp_op=4,cross_ts=True,ex=False,(cube_term=True) is the discrete convolution equivalent of caag
      aep8,_,_ = num_Fs(U_i,P_i,theta_i,
                        layout,layout,
                        turb,
                        K,
                        Ct_op=3,wav_Ct=wav_Ct,
                        Cp_op=4,
                        cross_ts=True,ex=False,cube_term=True)
            
      _,Fourier_coeffs_noCp_PA = simple_Fourier_coeffs((P_i*U_i*len(P_i))/(2*np.pi))
      aep9,_ = caag_PA(Fourier_coeffs_noCp_PA,
                 layout,layout,
                 turb,
                 K,
                 wav_Ct,
                 RHO=1.225)

      print(f'with {str(m*36)} bins:')
      print("num_F: {}".format(np.sum(aep8)))
      print("caag : {}".format(np.sum(aep9)))

print("the two *should* converge ")

#%% other small details 
Ct = 0.5
ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
print("ep: {}".format(ep))
print("hand result: 0.219736")


#%%
#%%This is my validation methodology 

# 1a. I construct a simple farm and find the aep by hand (hand calcs should be attached). The layout is constructed in such a way that if I made a mistake in the coordinate transform (which I found the hardest process to understand) then the aep would be different
# 1b. I write some simple functions that find the aep 
# 1c. Then check that num_Fs is giving the same result
# 2a. num_Fs with certain settings implements ntag ("Gaussian FLOWERS") numerically. I check that for the simple wind layout in 1a the two results converge. (I still need to do this by hand)
# 2b. for a "realistic" wind rose (using floris to parse a wind rose data folder), I check that numF agrees with ntag.
# 3a. num_Fs with certain settings implements caag (Gaussian equivalent of "Jensen FLOWERS") numerically. I check that for the simple wind layout in 1a the two results converge. (I still need to do this by hand)
# 3b. for a "realistic" wind rose (using floris to parse a wind rose data folder), I check that numF agrees with caag.

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
# coords: (0,0) (-0.2,-3) (+0.2,-3) (in rotor diameter)
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
# The only mistake in coordinate system would be theta = -theta + 180 (this would produce the same power). 
# Since in both cases the power generation is the same (note that 2. would be 0.5*P_a + 0.5*P_c if the first coordinate system was wrong), the coordinate system must be correct
# 
# The power calculations were performed 1) by hand 2) using simplified functions 3) using the actual functions. They all agree


#%%
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

U_inf = 15
K = 0.03

def new_wr1(no_bins):
    if not no_bins%4 == 0:
        raise ValueError("no_bins must be a multiple of 4") 
    theta_i = np.linspace(0,2*np.pi,no_bins,endpoint=False)
    case = 3
    if case == 1: 
        print("impulse wr")
        U_i = np.zeros(no_bins)
        U_i[0] = U_inf
        P_i = np.zeros(no_bins)
        P_i[0] = 1.0        
    elif case == 2: 
        print("uniform wr")
        U_i = np.full(no_bins,U_inf)  
        P_i = np.full(no_bins,1/(no_bins)) 
    elif case == 3: 
        print("two directions wr")
        U_i = np.zeros(no_bins)
        U_i[0],U_i[no_bins//4] = U_inf,13
        P_i = np.zeros(no_bins)
        P_i[0],P_i[no_bins//4] = 0.5,0.5
    elif case == 4: 
        print("east impulse")
        U_i = np.zeros(no_bins)
        U_i[no_bins//4] = U_inf
        P_i = np.zeros(no_bins)
        P_i[no_bins//4] = 1.0
    return U_i, P_i, theta_i

no_bins = 640
U_i,P_i,theta_i = new_wr1(no_bins)

def rotate_coordinates(coords, angle):
    # Convert angle to radians
    theta = np.deg2rad(-angle)
    # Rotation matrix for clockwise rotation
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    
    # Rotate coordinates
    rotated_coords = np.dot(coords, R.T) 
    return rotated_coords

layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))
#layout = np.array(((0,0),(0,-3),(0.4,-3)))
layout = rotate_coordinates(layout,0)

from utilities.helpers import simple_Fourier_coeffs,get_WAV_pp
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)
wav_ep = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))
lim = (np.sqrt(wav_Ct/8)-wav_ep)/K # the x limit
print("lim: {}".format(lim))
from utilities.AEP3_functions import num_Fs,ntag_PA
#this should agree with num_F
pow_j1,_,_ = num_Fs(U_i,P_i,theta_i,
                    layout,layout,
                    turb,
                    K,
                    Ct_op=3,wav_Ct = wav_Ct, #cnst Ct
                    Cp_op=2, #global Cp
                    cross_ts=False,cube_term=False,ex=False)

c,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/((2*np.pi)))
A_n, Phi_A  = cjd3_PA_terms

pow_j2,_ = ntag_PA(cjd3_PA_terms,
            layout,layout,
            turb,
            K,
            wav_Ct,
            u_lim=1.9)

print("pow_j1: {}".format(pow_j1))
print("pow_j2: {}".format(pow_j2))
print("=== sums ===")
print("sum1: {}".format(np.sum(pow_j1)))
print("sum2: {}".format(np.sum(pow_j2)))

#%% construct the Fourier?
def f(cjd3_PA_terms,x):
    A_n, Phi_A = cjd3_PA_terms
    n = np.arange(0,len(A_n),1)
    return np.sum(A_n[None,:]*np.cos(n[None,:]*x[:,None]+Phi_A[None,:]),axis=-1)

xs = np.linspace(0,2*np.pi,10000)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.plot(xs,f(cjd3_PA_terms,xs))
ax.set(xlim=(-0.1,2*np.pi+0.1))
#%%
a_0,a_n,b_n = c
A_n = np.sqrt(a_n**2+b_n**2)
Phi_n = -np.arctan2(b_n,a_n)
n = np.arange(1,len(A_n)+1,1)
xs = np.linspace(-0.1,2*np.pi+0.1,10000,endpoint=False)
yr = a_0/2 + np.sum(A_n[None,:]*np.cos(n[None,:]*xs[:,None]+Phi_n[None,:]),axis=-1)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.plot(xs,yr)
ax.vlines(theta_i[0],0,np.max(yr))
ax.vlines(theta_i[no_bins//4],0,np.max(yr),color='orange')

#%%
import numpy as np
from scipy.fft import fft

# Discretize the function
N = no_bins  # Number of samples
x = np.linspace(0, 2*np.pi, N, endpoint=False)
y = turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/((2*np.pi))

# Compute the Fourier coefficients
coeffs = fft(y) / N

# Extract a_n and b_n for n=0 to N/2
a = 2 * coeffs.real
b = -2 * coeffs.imag

# Compute A_n and Phi_n
A_n = np.sqrt(a**2 + b**2)
Phi_n = np.arctan2(-b, a)

# Now, you can use A_n and Phi_n to reconstruct the function
g = np.zeros_like(x)
for n in range(N//2):  # Only consider up to N/2 due to Nyquist theorem
    g += A_n[n] * np.cos(n * x + Phi_n[n])

# Plot the original and reconstructed functions
import matplotlib.pyplot as plt
plt.plot(x, y, label='Original f(x)')
plt.plot(x, g, '--', label='Reconstructed g(x)')
plt.legend()
plt.show()


#%%
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.scatter(layout[:,0],layout[:,1])
#ax.set(aspect='equal')
for i in range(len(layout[:,0])):
       ax.annotate(str(i),(layout[i,0],layout[i,1]))

#%% 
from utilities.helpers import fixed_rectangular_domain, deltaU_by_Uinf_f, find_relative_coords
layout3 = np.array(((0,0,)))
X,Y,plot_points = fixed_rectangular_domain(10,200)
import matplotlib.pyplot as plt
from matplotlib import cm
R,THETA = find_relative_coords(layout,plot_points)
THETA = THETA + np.pi
THETA = np.mod(THETA + np.pi, 2 * np.pi) - np.pi
fig,ax = plt.subplots(figsize=(3,3),dpi=200)
cf = ax.contourf(X,Y,THETA[:,2].reshape(X.shape),50,cmap=cm.coolwarm)
fig.colorbar(cf)

#%%
from ipywidgets import *

def update(theta_i=0):
      THETA1 = np.arctan2(X,Y) - theta_i
      Z = deltaU_by_Uinf_f(R.reshape(-1),THETA1.reshape(-1),wav_Ct,0.03,None,True)
      fig,ax = plt.subplots(figsize=(3,3),dpi=200)
      cf = ax.contourf(X,Y,Z.reshape(X.shape),50,cmap=cm.coolwarm)
      fig.colorbar(cf)

interact(update, theta_i= widgets.FloatSlider(value=0, min=-4*np.pi, max=4*np.pi, step=np.pi/4) )

#%%
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
cf = ax.contourf(X,Y,THETA2,50,cmap=cm.coolwarm)
fig.colorbar(cf)

#%%
