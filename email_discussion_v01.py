#%% This is a script to summarise the email discussion.

%load_ext autoreload
%autoreload 2

from AEP3_3_functions import num_F_v02,rectangular_layout,get_floris_wind_rose,get_WAV
import numpy as np

from turbines_v01 import iea_10MW
turb = iea_10MW()
U_i,P_i = get_floris_wind_rose(6)
theta_i = np.linspace(0,360,72,endpoint=False)
layout = rectangular_layout(7,7,0)

WAV_CT = get_WAV(U_i,P_i,turb,turb.Ct_f)
WAV_CT = np.sum(turb.Cp_f(U_i)*P_i)
#in the first tests set power coefficient to 1
turb.Cp_f = lambda x: np.ones_like(x)
a=None
def simple_aep(Ct_op=1,Cp_op=1,cross_ts=True,ex=True,cube_term=True):
    global a
    pow_j,_,a= num_F_v02(U_i,P_i,np.deg2rad(theta_i),
                    layout,
                    layout,
                    turb,
                    K=0.025,
                    u_lim=None,
                    Ct_op=Ct_op,WAV_CT=WAV_CT,
                    Cp_op=Cp_op,WAV_CP=None,
                    cross_ts=cross_ts,ex=ex,cube_term=cube_term)
    return np.sum(pow_j)

f1 = simple_aep()
f2 = simple_aep(cross_ts=False)
f3 = simple_aep(Cp_op=4)
f4 = simple_aep(Cp_op=5)

#%% the illustration
sites = []
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.bar(np.deg2rad(theta_i),U_i*P_i,color='grey',linewidth=1,width=2*np.pi/72)
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')