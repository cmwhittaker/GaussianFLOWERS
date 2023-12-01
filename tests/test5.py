#%% Checking the Jensen FLOWERS implementation

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

# First test: Is the wake region correct?
K=0.05 #wake expansion rate

U_inf1 = 8 #15
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

def Uw(U_i,du):
    return U_i*(1-du)
#%%
U_i = 10
import matplotlib.pyplot as plt

from matplotlib import cm 
# (theta<=theta_c)|(-theta_c<=theta)
x = np.linspace(-20,20,1000)

y = K*x+0.5
xx, yy = np.meshgrid(np.linspace(-20,20,1000),np.linspace(-20,20,1000))
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
du = U_delta_J2(U_i,xx,yy)
du = np.where(np.sqrt(xx**2+yy**2)>1,du,0)
cf = ax.contourf(xx,yy,Uw(U_i,du),50,cmap=cm.coolwarm)
fig.colorbar(cf)
ax.plot(x,y,color='black')


#%% test 5B

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
print("simple_aep: {}".format(simple_aep))
#then this is useful for debugging
norm_DU_total = np.array((DU_nT0T1,DU_nT0T1,DU_nT2,DU_nT3)) + np.array((DU_eT1T2T3,DU_eT1T2T3,DU_eT1T2T3,DU_eT0))
#%% and as a sanity check:
alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst
no_wake_aep = 4*np.sum(alpha*turb.Cp_f(Ui_wav)*Ui_wav**3)
#%% this should agree with a more complex wake function
# my "own" flowers function
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

from utilities.helpers import simple_Fourier_coeffs,trans_bearing_to_polar,find_relative_coords
# =====================================
no_bins = 4*100# 4*30000 seems to be the max
# ====================================
U_WB_i2, P_WB_i2,theta_WB_i2 = new_wr1(no_bins)
theta_WB_i2 = theta_WB_i2

layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))

from utilities.flowers_interface import FlowersInterface
flower_int = FlowersInterface(U_WB_i2,P_WB_i2,np.rad2deg(theta_WB_i2+np.pi), layout, turb,num_terms=no_bins//2, k=K) 
flower_aep = np.sum(flower_int.calculate_aep())

U_i2,P_i2,theta_i2 = trans_bearing_to_polar(U_WB_i2,P_WB_i2,theta_WB_i2)

Fourier_coeffs1_Ct0,_ = simple_Fourier_coeffs(U_i2*P_i2*len(P_i2)/(2*np.pi)) 
c_0,_,_ = Fourier_coeffs1_Ct0 

Fourier_coeffs1_Ct,_ = simple_Fourier_coeffs((1 - np.sqrt(1 - turb.Ct_f(U_i2))) * U_i2*P_i2*len(P_i2)/(2*np.pi)) 
a_0,a_n,b_n = Fourier_coeffs1_Ct

a_0,a_n,b_n = Fourier_coeffs1_Ct
r_jk,theta_jk = find_relative_coords(layout,layout) 
R = r_jk
THETA = theta_jk 

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
m = np.arange(1, len(a_n)+1)
R = R[:, :, None]
THETA = THETA[:, :, None] 
theta_c = theta_c[:, :, None] 

# Vectorized contribution of higher Fourier modes
du += np.sum(2*(1 / (m * (2 * K * R + 1)**2) * (
    a_n * np.cos(m*THETA) + b_n * np.sin(m*THETA)) * (
        np.sin(theta_c * m) + 2 * K * R / (m**2 * (2 * K * R + 1)) * (
            ((theta_c * m)**2 - 2) * np.sin(theta_c * m) + 2*theta_c*m *np.cos(theta_c * m)))), axis=2)

# Apply mask for points within rotor radius
du = np.where(mask_area,a_0,du)
np.fill_diagonal(du, 0.)
# Sum power for each turbine
du = np.sum(du, axis=1) #superposistion sum
wav = (c_0*np.pi - du)
alpha = turb.Cp_f(wav)*wav**3 #turbine sum #np.sum((u0 - du)**3) 
aep = (0.5*turb.A*1.225*alpha)/(1*10**6)

print("simple_aep: {}".format(simple_aep))
print("my_aep    : {}".format(np.sum(aep)))
print("michael   : {}".format(flower_aep))

#%%



mask = np.where(r_jk<=0.5,1,0) #true within radius



r = 2 * r_jk #because the formulas are all normalised by radius

theta_c = np.arctan(
        (1 / (2*r_jk) + K * np.sqrt(1 + K**2 - (2*r_jk)**(-2)))
        / (-K / (2*r_jk) + np.sqrt(1 + K**2 - (2*r_jk)**(-2)))
        )
theta_c = np.nan_to_num(theta_c)

r = 2 * r_jk  # Update according to the provided instruction
# Calculate the first term outside the summation
first_term = (a_0 * theta_c * (K * r * (theta_c**2 + 3) + 3)) / (3 * (K * r + 1)**3)

m = np.arange(1, len(a_n)+1)[None,None,:] #a_n includes a_0
a_n = a_n[None, None,:] 
b_n = b_n[None, None,:] 
r = r[:, :, None]
theta_jk = theta_jk[:, :, None] 
theta_c = theta_c[:, :, None] 

# Calculate the cosine and sine terms
cos_terms = np.cos(m * theta_jk)
sin_terms = np.sin(m * theta_jk)
cos_theta_c_terms = np.cos(m * theta_c)
sin_theta_c_terms = np.sin(m * theta_c)

# Calculate each term in the summation
term1 = 2 * (a_n * cos_terms + b_n * sin_terms) / (m * (K * r + 1)**3)
term2 = sin_theta_c_terms * (m**2 * (K * r * (theta_c**2 + 1) + 1) - 2 * K * r)
term3 = 2 * m * theta_c * K * r * cos_theta_c_terms
# Calculate the summation terms
summation_terms = np.sum(term1 * (term2 + term3),axis=2)
print("summation_terms.shape: {}".format(summation_terms.shape))

# Sum the summation terms and add to the first term
delta_U1 = first_term + summation_terms
delta_U = np.where(mask,0,delta_U1)
delta_U = np.sum(delta_U1,axis=1)
print("delta_U.shape: {}".format(delta_U.shape))   
print("delta_U: {}".format(delta_U))

wav = c_0*np.pi - delta_U 
print("wav.shape: {}".format(wav.shape))

alpha = turb.Cp_f(wav)*wav**3 #turbine sum #np.sum((u0 - du)**3) 
aep = (0.5*turb.A*1.225*alpha)/(1*10**6)
    
# Fourier_coeffs1_Ct = np.array(flower_int.fs['a']*turb.U)[0],np.array(flower_int.fs['a']*turb.U)[1:],np.array(flower_int.fs['b']*turb.U)[1:]

print("simple_aep: {}".format(simple_aep))
print("my_aep    : {}".format(np.sum(aep)))
print("michael   : {}".format(flower_aep))

#%%
a = flowers_C(Fourier_coeffs1_Ct,c_0,layout,layout,turb,K,u_lim=0.5,RHO=1.225)







#%%
ab = flower_int.fs*turb.U
print(ab)
print(Fourier_coeffs1_Ct)


#%%



alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst
#power is calculated from weight-averaged velocities
simple_aep = alpha*turb.Cp_f(U_wav)*U_wav**3

#%%
layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(3,3),dpi=200)
ax.set(aspect='equal')
ax.scatter(layout[:,0],layout[:,1])
for i in range(layout.shape[0]):
    ax.annotate(str(i),(layout[i,0],layout[i,1]))#%%
#%%

#%%





# === North @ U_inf1 ===
#T0, T1 are unwaked
#%%

xx, yy = np.meshgrid(np.linspace(-10,10,1000),np.linspace(-10,10,1000))
r = np.sqrt(xx**2+yy**2)
theta = np.arctan2(yy,xx)
theta_c = np.arctan(
            (1 / (2*r) + K * np.sqrt(1 + K**2 - (2*r)**(-2)))
            / (-K / (2*r) + np.sqrt(1 + K**2 - (2*r)**(-2)))
            ) 
theta_c = np.nan_to_num(theta_c)
theta_c = np.where(r<0.5,0,theta_c)
import matplotlib.pyplot as plt
from matplotlib import cm 
# (theta<=theta_c)|(-theta_c<=theta)
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
a = np.where( (-theta_c<=theta)&(theta<=theta_c),1,0)
cf = ax.contourf(xx,yy,U_delta_J(xx,yy,10),50,cmap=cm.coolwarm)
fig.colorbar(cf)