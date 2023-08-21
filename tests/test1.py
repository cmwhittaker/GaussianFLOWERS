#%%
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
U_inf = 13 #can adjust, be careful of lim being <0.4
ct = turb.Ct_f(U_inf) #Ct
ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))
lim = (np.sqrt(ct/8)-ep)/K # the x limit
if lim < 0.01:
    lim = 0.01 #stop self produced wake
alpha = ((0.5*1.225*turb.A)/(1*10**6)) #turbine cnst

#simple implementation of the Gaussian wake model
def U_delta(x,y):
      U_delta = (1-np.sqrt(1-(ct/(8*(K*y+ep)**2))))*(np.exp(-x**2/(2*(K*y+ep)**2)))
      U_delta = np.where(y>lim,U_delta,0)
      return U_delta
def Pwr(U): #power given local wake velocity
     return alpha*turb.Cp_f(U)*U**3
#1xundisturbed, 2xsingularly waked
P_a = Pwr(U_inf)+2*Pwr(U_inf*(1-U_delta(0.2,3)))
#2xundisturbed, 1xsingularly waked
P_b = 2*Pwr(U_inf) +Pwr(U_inf*(1-U_delta(0,0.4)))
#2xundisturbed, 1xdoubly waked
P_c = 2*Pwr(U_inf) +Pwr(U_inf*(1-2*U_delta(0.2,3)))
#total power if there are no rotation mistakes
P_t = P_a + P_b

#Next check num_Fs is giving the same result
U_i = np.array((U_inf,U_inf))
P_i = np.array((0.5,0.5))
theta_i = np.array((0,np.pi/2))
layout1 = np.array(((0,0),(-0.2,-3),(+0.2,-3)))
layout2 = np.array(((0,0),(-3,-0.2),(-3,+0.2)))
#space two farms by 10000 diameters (so they don't intefere)
layout2[:,0] = layout2[:,0] + 10000
#put two farms together
layout = np.concatenate((layout1,layout2),axis=0)

#local thrust coeff (Ct_op=1),local power coeff (Cp_op=1), include cross terms (cross_ts=True), exact wake deficit (ex=True)
from utilities.AEP3_functions import num_Fs
pow_j,_,_ = num_Fs(U_i,P_i,theta_i,
                   layout,layout,
                   turb,
                   K,
                   Ct_op=1,
                   Cp_op=1,
                   cross_ts=True,ex=True)

print(f"hand power check aep:   {49.25:.2f}   (this is fixed)")
print(f"simple power check aep: {P_t:.4f}")
print(f"num_F power check aep:  {np.sum(pow_j):.4f}")

#%%
from utilities.helpers import find_relative_coords
Xt,Yt = np.array((0,0))
a1 = np.column_stack((Xt,Yt))
xt,yt = np.array((1,1))
b1 = np.column_stack((xt,yt))
theta_i = 0.2

Rt = np.sqrt((Xt-xt)**2+(Yt-yt)**2)
THETAt = np.pi/2 - np.arctan2(Yt-yt,Xt-xt) - theta_i
print(Rt,THETAt)
a,b = find_relative_coords(b1,a1) 
b = b - theta_i
print(a,b)