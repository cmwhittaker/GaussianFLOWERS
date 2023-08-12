#%% Figure to show the effect of changing the number of Fourier terms on runtime and accuracy
#use the middle / middle site / wind rose combo for now?

SPACING = 7
K = 0.03

import numpy as np
def rectangular_layout(no_xt,s,rot):
    low = (no_xt)/2-0.5
    xt = np.arange(-low,low+1,1)*s
    yt = np.arange(-low,low+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)
    Xt,Yt = [_.reshape(-1) for _ in [Xt,Yt]]
    rot_Xt = Xt * np.cos(rot) + Yt * np.sin(rot)
    rot_Yt = -Xt * np.sin(rot) + Yt * np.cos(rot) 
    layout = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
    return layout#just a single layout for now

from floris.tools import WindRose
def get_floris_wind_rose(site_n):
    fl_wr = WindRose()
    folder_name = "WindRoseData_D/site" +str(site_n)
    fl_wr.parse_wind_toolkit_folder(folder_name,limit_month=None)
    wr = fl_wr.resample_average_ws_by_wd(fl_wr.df)
    wr.freq_val = wr.freq_val/np.sum(wr.freq_val)
    U_i = wr.ws
    P_i = wr.freq_val
    return np.array(U_i),np.array(P_i)

import time
def floris_timed_aep(U_i,P_i,theta_i,layout,turb,wake=True,timed=True):
    from floris.tools import FlorisInterface
    fi = FlorisInterface("floris_settings.yaml")
    fi.reinitialize(wind_directions=theta_i, wind_speeds=U_i, time_series=True,layout_x=turb.D*layout[:,0],layout_y=turb.D*layout[:,1])
    if wake:
        if timed:
            timings = %timeit -o -q fi.calculate_wake()
            time = timings.best
        else:
            fi.calculate_wake()
            time = np.NaN
    else:
        fi.calculate_no_wake()
        time = np.NaN
    aep_array = fi.get_turbine_powers() #happy days, we can go from here
    pow_j = np.sum(P_i[:,None,None]*aep_array,axis=0)
    return pow_j/(1*10**6),time

def analytical_timed_aep(Fourier_coeffs_PA,layout,WAV_CT,K,turb,timed=True):
    from AEP3_3_functions import ntag_PA_v03
    if timed:
        result = []
        timings = %timeit -o -q result.append(ntag_PA_v03(Fourier_coeffs_PA,layout,layout,turb,WAV_CT,K))
        pow_j,_ = result[0]
        time = timings.best
    else:
        pow_j,_ = ntag_PA_v03(Fourier_coeffs_PA,layout,layout,turb,WAV_CT,K)
        time = np.NaN
    return pow_j,time

from turbines_v01 import iea_10MW
turb = iea_10MW()

from AEP3_3_functions import simple_Fourier_coeffs_v01
NO_BINS = 72
SITE = 6
U_i,P_i = get_floris_wind_rose(SITE)
WAV_CT = np.sum(turb.Ct_f(U_i)*P_i)
theta_i = np.linspace(0,360,NO_BINS,endpoint=False)
_,FULL_Fourier_coeffs_PA = simple_Fourier_coeffs_v01(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))
a_0,A_n,Phi_n = FULL_Fourier_coeffs_PA

layout = rectangular_layout(6,SPACING,rot=0)

powj_a,time_a = floris_timed_aep(U_i,P_i,theta_i,layout,turb,timed=True)

Nterms = np.arange(NO_BINS/2,0,-4).astype(int).tolist() + [3,2,1,0]
#np.arange(NO_BINS/2,0,-4).astype(int).tolist() + 
print("Nterms: {}".format(Nterms))

powj_b = np.zeros((len(Nterms),layout.shape[0]))
time_b = np.zeros((len(Nterms)))

for i in range(len(Nterms)): #for each number of terms
    #truncate the Fourier series
    Fourier_coeffs_PA = a_0, A_n[:Nterms[i]],Phi_n[:Nterms[i]]
    #analytical aep
    powj_b[i,:],time_b[i] = analytical_timed_aep(Fourier_coeffs_PA,layout,WAV_CT,K,turb,timed=True)
    print(f"Terms: {Nterms[i]} ({i+1}/{len(Nterms)})", end='\r')

#%% now plot the results
def pce_f(exact,approx):
    return 100*(exact-approx)/exact

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(2.6,2.6),dpi=200)
aep1 = np.sum(powj_a)
aep2 = np.sum(powj_b,axis=1)
ref_pce = pce_f(aep1,aep2[0])
pce = pce_f(aep1,aep2)
ax.scatter(Nterms,pce,color='blue')
ax.set(ylim=[None,None])
ax2 = ax.twinx()
ax2.scatter(Nterms,time_a/time_b,color='orange') #relative run time