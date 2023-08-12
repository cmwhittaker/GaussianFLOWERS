#%% need to work out average wind speed binned floris
#the floris way

%load_ext autoreload
%autoreload 2

import numpy as np
from floris.tools import FlorisInterface, WindRose
fl_wr = WindRose()
site_n = 6
folder_name = "WindRoseData_D/site" +str(site_n)
fl_wr.parse_wind_toolkit_folder(folder_name,limit_month=None)
wr = fl_wr.resample_average_ws_by_wd(fl_wr.df)
#%
# theta_i = np.array(((0,90)))
# P_i = np.array(((0.5,0.5,)))
# U_i = np.array(((10,10,)))

wr.freq_val = wr.freq_val / np.sum(wr.freq_val)
P_i = np.array(wr.freq_val)
U_i = np.array(wr.ws)
theta_i = np.array(wr.wd)

from turbines_v01 import iea_10MW
turb = iea_10MW()
Cp_f = turb.Cp_f
D = turb.D
RHO = 1.225
A = turb.A
NO_BINS = 72

layout = np.array(((0,0),(5,0),(10,0)))
fi = FlorisInterface("floris_settings.yaml")
fi.reinitialize(wind_directions=theta_i, wind_speeds=U_i, time_series=True,layout_x=D*layout[:,0],layout_y=D*layout[:,1])
fi.calculate_wake()
a = fi.get_turbine_powers() #happy days, we can go from here
pow_j = np.sum(P_i[:,None,None]*a,axis=0)
aep1 = np.sum(pow_j)/(1*10**6)
print("aep1: {}".format(aep1))

# what about flow field visualisation? leave this for later?
#%% no wake, but using the floris wind rose 
aep2 = layout.shape[0]*np.sum(0.5*P_i*turb.A*1.225*Cp_f(U_i)*U_i**3)/(1*10**6)
print("aep2: {}".format(aep2))

pce = 100*np.abs(aep1-aep2)/aep1
print("pce: {}".format(pce))

#%% no wake, using my own wind rose
import numpy as np
from distributions_vC05 import wind_rose

theta_i = np.linspace(0,2*np.pi,NO_BINS,endpoint=False)
def get_own_wind_rose(site_n):
    #my own wind rose 
    own_wr = wind_rose(bin_no_bins=NO_BINS,custom=None,a_0=8,site=site_n,Cp_f=turb.Cp_f)
    return own_wr

ow_wr = get_own_wind_rose(site_n)
U_i, P_i = ow_wr.avMagnitude,ow_wr.frequency
aep2 = np.sum(0.5*A*RHO*Cp_f(U_i)*U_i**3)/(1*10**6)
print("aep2: {}".format(aep2))

#%% now need to "validate" the analyitcal way (but using a floris wind rose...)

def simple_Fourier_coeffs(data):   
    #naively fit a Fourier series to data (no normalisation takes place (!))
    import scipy.fft
    c = scipy.fft.rfft(data)/np.size(data)
    length = np.size(c)-1 #because the a_0 term is included in c # !! 
    a_0 = 2*np.real(c[0])
    a_n = 2*np.real(c[-length:])
    b_n =-2*np.imag(c[-length:])
    Fourier_coeffs = a_0,a_n,b_n
    # #convert to phase amplitude form
    A_n = np.sqrt(a_n**2+b_n**2)
    Phi_n = -np.arctan2(b_n,a_n)
    Fourier_coeffs_PA = a_0,A_n,Phi_n
    
    return Fourier_coeffs,Fourier_coeffs_PA

def plot_from_layout(turb,layout,border,res):
    b = border+1
    xt,yt = layout[:,0],layout[:,1]
    xp = np.linspace(np.min(xt)-b,np.max(xt)+b,res)
    yp = np.linspace(np.min(yt)-b,np.max(yt)+b,res)
    X,Y = np.meshgrid(xp,yp)
    plot_points = np.column_stack((X.reshape(-1),Y.reshape(-1)))
    return X,Y,plot_points


Fourier_coeffs,Fourier_coeffs_PA = simple_Fourier_coeffs(Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))

WAV_CT = np.sum(turb.Ct_f(U_i)*P_i)
K = 0.03

from AEP3_3_functions import ntag_v02,ntag_PA_v03,gen_local_grid_v01C
X,Y,plot_points = plot_from_layout(turb,layout,10,300)
r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
a3,b,aep3 = ntag_v02(r_jk,theta_jk,Fourier_coeffs,WAV_CT,K,turb.A,rho=1.225)
print("aep3: {}".format(aep3))

a4,b,aep4 = ntag_PA_v03(Fourier_coeffs_PA,layout,layout,WAV_CT,K,turb.A,RHO=1.225)
print("aep4: {}".format(aep4))

#%%
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
cf = ax.contourf(X,Y,a4.reshape(X.shape),50,cmap=cm.coolwarm)
fig.colorbar(cf)
ax.set(aspect='equal')
#IT WORKS
#%%
from distributions_vC05 import wind_rose
own_wr = wind_rose(bin_no_bins=NO_BINS,custom=None,a_0=8,site=site_n,Cp_f=turb.Cp_f)
own_wr.cjd3_PA_all_coeffs
own_wr.cjd3_full_Fourier_coeffs

#%%
print("github test4")