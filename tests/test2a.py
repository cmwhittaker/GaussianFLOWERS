#%% test3 which is the 15ms ntag rotation check
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
alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst#
U_inf = 13
U_i = np.array((U_inf,))
P_i = np.array((1.0,))
dir_m = 2
theta_i = np.array((dir_m*np.pi/2,))
from utilities.helpers import get_WAV_pp,simple_Fourier_coeffs
#weight average using expected power production
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)
wav_ep = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))
lim = (np.sqrt(wav_Ct/8)-wav_ep)/K # the x limit
print("lim: {}".format(lim))
#print("limit: {}".format(lim))
if lim < 0.01:
    lim = 0.01 #stop self produced wake

#next check if num_F and ntag are giving the same result
#Next check num_Fs is giving the same result
layout = np.array(((0,0),(-0.2,-3),(+0.2,-3)))
from utilities.helpers import fixed_rectangular_domain
X,Y,plot_points = fixed_rectangular_domain(-7,200)
# num_F with the following assumptions:
# cnst thrust coeff (Ct_op=3, wav_Ct=wav_Ct),global power coeff (Cp_op=2), neglect cross terms (cross_ts=False), approx. wake deficit (ex=False), neglect cube terms (cube_term=False)
from utilities.AEP3_functions import num_Fs,ntag_PA
pow_j1,_,Uff = num_Fs(U_i,P_i,theta_i,
                    layout,plot_points,
                    turb,
                    K,
                    Ct_op=3,wav_Ct=wav_Ct,
                    Cp_op=2,
                    cross_ts=False,ex=False,cube_term=False)
print("powj_1: {}".format(np.sum(pow_j1)))

a = num_Fs.DUt_ijk
b = num_Fs.DUff_ijk

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
#ax.set(aspect='equal')
ax.contourf(X,Y,b[0,:,2].reshape(X.shape))
ax.scatter(layout[:,0],layout[:,1],s=20,marker='x',c='black')
#%%
a = num_Fs.DUff_ijk

#this needs to agree with ntag

def new_wr1(no_bins):
      #needs large amount of bins for narrow spikes in the Fourier series
      indx = dir_m*no_bins//4
      if not no_bins%4 == 0:
            raise ValueError("no_bins must be a multiple of 4") 
      U_i2 = np.zeros(no_bins)
      U_i2[indx] = U_inf  #0 and pi/2 bin
      P_i2 = np.zeros(no_bins)
      P_i2[indx] = 1.0
      return U_i2, P_i2

U_i2,P_i2 = new_wr1(400*4) 
_,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i2)*(P_i2*(U_i2**3)*len(P_i2))/((2*np.pi)))

pow_j2,_ = ntag_PA(cjd3_PA_terms,
            layout,layout,
            turb,
            K,
            wav_Ct)

print("===== Test2a ======")
print(f"num_F power check aep:         {np.sum(pow_j1):.6f}")
print(f"ntag power check aep:          {np.sum(pow_j2):.6f}")

#%%



def new_wr1(no_bins):
            if not no_bins%4 == 0:
                  raise ValueError("no_bins must be a multiple of 4") 
            theta_i = np.linspace(0,2*np.pi,no_bins,endpoint=False)
            U_i = np.zeros(no_bins)
            #U_i[0], U_i[no_bins//4] = 15,13  #0 and pi/2 bin
            U_i[no_bins//4] = 13 #0 and pi/2 bin
            P_i = np.zeros(no_bins)
            #P_i[0], P_i[no_bins//4] = 0.5, 0.5
            P_i[no_bins//4]= 1.0
            return U_i, P_i, theta_i

pow_j1,_,_ = num_Fs(U_i,P_i,theta_i,
                    layout,layout,
                    turb,
                    K,
                    Ct_op=3,wav_Ct=wav_Ct,
                    Cp_op=2,
                    cross_ts=False,ex=False,cube_term=False)

def f(no_bins):
      #for ntag, a larger number of bins is needed so that the Fourier series 
      #approximation of the wind rose converges to two dirac delta spikes
      #this does not effect the result
      
      U_i2,P_i2,theta_i2 = new_wr1(no_bins) #add more bins
      _,cjd3_PA_terms = simple_Fourier_coeffs(turb.Cp_f(U_i2)*(P_i2*(U_i2**3)*len(P_i2))/((2*np.pi)))

      pow_j1,_ = ntag_PA(cjd3_PA_terms,
                  layout,layout,
                  turb,
                  K,
                  wav_Ct)
      
      return np.sum(pow_j1)

print("===== Test3 ======")
print(f"hand power check aep:          {22.21:.2f} (this is fixed)")
print(f"simple power check aep:        {P_t:.6f}")
print(f"num_F power check aep:         {np.sum(pow_j1):.6f}")
print("+++ convergence test +++")
print(f"ntag {72}   bin power check aep: {f(72):.6f}")
print(f"ntag {360}  bin power check aep: {f(360):.6f}")
print(f"ntag {1440} bin power check aep: {f(1440):.6f}")
#%%

_,(a,b) = simple_Fourier_coeffs(turb.Cp_f(U_i2)*(P_i2*(U_i2**3)*len(P_i2))/((2*np.pi)))

#%%
# is the fourier series correct ?!?
# (yes)
xs = np.linspace(0,2*np.pi,100,endpoint=False)
ys = 2*xs+1
Fourier_coeffs,PA_Fourier_coeffs = simple_Fourier_coeffs(ys)
# reconstruct PA
a_0,A_n,Phi_n = PA_Fourier_coeffs
n = np.arange(1,A_n.size+1,1)
yr1 = a_0/2 + np.sum(A_n[None,:]*np.cos(n[None,:]*xs[:,None]+Phi_n[None,:]),axis=1)
# recontruct normal
a_0,a_n,b_n = Fourier_coeffs
n = np.arange(1,A_n.size+1,1)
yr2 = a_0/2 + np.sum(a_n[None,:]*np.cos(n[None,:]*xs[:,None])+b_n[None,:]*np.sin(n[None,:]*xs[:,None]),axis=1)
# plot result
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.scatter(xs,ys,marker='x')
ax.scatter(xs,yr1)
ax.scatter(xs,yr2)

#%%
def simple_Fourier_coeffs_v2(data):   
    # naively fit a Fourier series to data (no normalisation takes place (!))
    # returns both sine/cosine and phase/amplitude coefficients
    # reconstruction uses the formula:
    # a_0/2 + (a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)
    # or 
    # a_0/2 + (A_n*np.cos(n_b*theta_b+Phi_n)
    import scipy.fft
    c = scipy.fft.rfft(data)/np.size(data)
    length = np.size(c)-1 #because the a_0 term is included in c # !! 
    a_0 = 2*np.real(c[0])
    a_n = 2*np.real(c[-length:])
    b_n =-2*np.imag(c[-length:])
    # #convert to phase amplitude form
    A_n = np.sqrt(a_n**2+b_n**2)
    print("A_n.shape: {}".format(A_n.shape))
    Phi_n = -np.arctan2(b_n,a_n)
    A_n = np.concatenate((np.array((a_0/2,)),A_n))
    print("A_n.shape: {}".format(A_n.shape))
    Phi_n = np.concatenate((np.array((0,)),Phi_n))
    Fourier_coeffs_PA = A_n,Phi_n
    
    return Fourier_coeffs_PA

xs = np.linspace(0,2*np.pi,100,endpoint=False)
ys = 2*xs+1
PA_Fourier_coeffs2 = simple_Fourier_coeffs_v2(ys)
A_n,Phi_n = PA_Fourier_coeffs2
n = np.arange(0,A_n.size,1)
yr = np.sum(A_n[None,:]*np.cos(n[None,:]*xs[:,None]+Phi_n[None,:]),axis=1)

# plot result
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.scatter(xs,ys,marker='x')
ax.scatter(xs,yr)


#%% untouched


def U_delta_SA(a,o):
      #"dumb" way to convert to polar
      #a:adjacent to angle, o:opposite to angle
      r = np.sqrt(a**2+o**2)
      theta = np.arctan2(a,o)
      #wake velocity deficit with small angle
      #use weight-averaged globals for Ct and ep
      U_delta = (1-np.sqrt(1-(wav_Ct/(8*(K*r*1+wav_ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+wav_ep)**2)))   
      U_delta = np.where(r>lim,U_delta,0)
      return U_delta

def Pwr_NC(U_inf,delta): 
     #power neglecting the cross terms (and without cubic term)
     #and with a global power coefficient
     U_cube = U_inf**3*(1-3*delta+3*delta**2)
     return alpha*turb.Cp_f(U_inf)*U_cube

#Northerly 15ms: 1xundisturbed, 2xsingularly waked
P_n = Pwr_NC(U_i[0],0)+2*Pwr_NC(U_i[0],U_delta_SA(0.2,3))
#Easterly 13ms: 2xundisturbed, 1xsingularly waked
P_e = 2*Pwr_NC(U_i[1],0)+Pwr_NC(U_i[1],U_delta_SA(0,0.4))
#total power
P_t = P_i[0]*P_n + P_i[1]*P_e
