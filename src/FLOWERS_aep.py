#%% setup cell
import numpy as np
import sys
import os
sys.path.append(os.path.join('..', 'src')) #allow to import from utilities (there may be a better way ...)
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2

import warnings

# Suppress all runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utilities.turbines import iea_10MW
turb = iea_10MW()

from utilities.helpers import get_floris_wind_rose
U_i,P_i,_,_ = get_floris_wind_rose(6)
thetaD_i = np.linspace(0,360,72,endpoint=False)
layout = np.array([[0,0]])

from utilities.AEP3_functions import LC_flowers_timed_aep
from utilities.helpers import adaptive_timeit
from pathlib import Path

pow_z,time_1 = LC_flowers_timed_aep(U_i,P_i,thetaD_i,layout,turb,0.05,timed=True)
aep1 = np.sum(pow_z) 

from utilities.helpers import simple_Fourier_coeffs,get_WAV_pp
_,Fourier_coeffs3_PA = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)

from utilities.plotting_funcs import si_fm
print(f'JFLOWERS: {aep1:.2f} in {si_fm(time_1)}')

#%% 
U_inf1 = 15
U_inf2 = 13
U_WB_i = np.array((U_inf1,U_inf2,))
P_WB_i = np.array((0.7,0.3,))
theta_WB_i = np.array((0,90,))
from utilities.flowers_interface import FlowersInterface
K = 0.05
flower_int = FlowersInterface(U_WB_i,P_WB_i,theta_WB_i, layout, turb,num_terms=2, k=K) 
print(np.sum(flower_int.calculate_aep())/10**6)

#%% my own function to do the flowers calculations

ct = self.turb.Ct_f(self.U_i)
cp = self.turb.Cp_f(self.U_i)

# Normalize wind speed by cut-out speed
nU_i = self.U_i/self.U #(new variable to allow multiple runnings without reinitalisation)

# Average freestream term
c = np.sum(cp**(1/3) * nU_i * self.P_i)

# Fourier expansion of wake deficit term
data = turb.Cp_f(U_i) * (1 - np.sqrt(1 - turb.Ct_f(U_i))) * P_i
from utilities.helpers import simple_Fourier_coeffs
Fourier_coeffs,_ = simple_Fourier_coeffs(data)
a_0,a_n,b_n = Fourier_coeffs
r_jk,theta_jk = find_relative_coords(layout1,layout2)

du = np.nansum((1 / (pi_m * (2 * self.k * R + 1)**2) * (
            a * np.cos(t_pi_m_T) + b * np.sin(t_pi_m_T)) * (
                s_t_pi_m_tc + 2 * self.k * R / (m**2 * (2 * self.k * R + 1)) * (
                    ((t_pi_m_tc)**2 - 2) * s_t_pi_m_tc + 2*t_pi_m_tc*np.cos(t_pi_m_tc)))), axis=2)

#%% need a function to sort
from utilities.helpers import get_floris_wind_rose
U_i,P_i,theta_i,fl_wr = get_floris_wind_rose(6)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(theta_i,U_i*P_i)
ax.set_theta_zero_location('E')


#%%
from utilities.helpers import get_floris_wind_rose
a,b,c = get_floris_wind_rose(5)

#%%
x = np.linspace(-10,10,300)
y = x
y1  = np.mod(x-np.pi,2*np.pi)-np.pi
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')
ax.plot(x,y)
ax.plot(x,y1)
#need to convert to the "standard coordinate system"
#%%
from utilities.AEP3_functions import ntag_PA
#ntag (No cross Terms Analytical Gaussian) (aep+time)
aep_func_d = lambda: ntag_PA(Fourier_coeffs3_PA,
                                    layout,
                                    layout,
                                    turb,
                                    0.03, 
                                    #(Ct_op = 3 cnst) 
                                    #(Cp_op = 2 global )    
                                    wav_Ct)
(powj_d,_),time_2 = adaptive_timeit(aep_func_d,timed=True)
aep2 = np.sum(powj_d)
print(f'GFLOWERS: {aep2:.2f} in {si_fm(time_2)}')
#%%

layout = np.array([[0,0],[3,0],[10,0]])



def get_sort_index(layout,theta_i):
    #sorts turbines from furthest upwind in wind orientated frame   

    def rotate_layout(layout,rot):
        #rotates layout anticlockwise by angle rot 
        Xt,Yt = layout[:,0],layout[:,1]
        rot_Xt = Xt * np.cos(rot) - Yt * np.sin(rot)
        rot_Yt = Xt * np.sin(rot) + Yt * np.cos(rot) 
        layout_r = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
        return layout_r       
    
    #from wind orientated frame, rotation is opposite
    layout_r = rotate_layout(layout,-theta_i)

    sort_index = np.argsort(layout_r[:, 0]) #sort index, with furthest upwind (<x) first
    return layout_r,sort_index

theta_i = 180
thetaR_i = np.deg2rad(-theta_i)

layout_r,sort_index = get_sort_index(layout,thetaR_i)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(xlim=(-15,15),ylim=(-15,15))
ax.scatter(layout_r[:,0],layout_r[:,1])
for i in range(3):
    ax.annotate(str(sort_index[i]),(layout_r[i,0],layout_r[i,1]))

#%%
#%%
indx = get_sort_index(layout,thetaR_i)
layout_n = layout[indx] #re-sort from upwind first

print(indx)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5),dpi=200)
ax.set(xlim=(-15,15),ylim=(-15,15))
ax.scatter(layout_n[:,0],layout_n[:,1])
ax.plot([0,-np.cos(thetaR_i)],[0,-np.sin(thetaR_i)])
for i in range(layout.shape[0]):
    ax.annotate(str(i),(layout_n[i,0],layout_n[i,1]))

#%%
names = ['t1:','t2:','t3:']
layout_sort = layout[indx]
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5),dpi=200)
ax.scatter(layout_sort[:,0],layout_sort[:,1])
ax.set_aspect('equal')
for i in range(layout.shape[0]):
    ax.annotate(names[i]+str(indx[i]),(layout[i,0],layout[i,1]))
#%%
ax.plot([0,-np.cos(thetaR_i)],[0,-np.sin(thetaR_i)])
labels = [str(_) for _ in indx]


import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5),dpi=200)
ax.set(aspect='equal')
ax.scatter(a,b)
#%%
from utilities.flowers_interface import FlowersInterface
a = FlowersInterface(1,2,3,4,5)

#%% Hijacking this program to test some floris stuff
site_n = 2
from pathlib import Path
current_file_path = Path(__file__)
folder_name = current_file_path.parent.parent/ "data" / "WindRoseData_D" / ("site"+str(site_n))
from floris.tools import WindRose
fl_wr = WindRose()
fl_wr.parse_wind_toolkit_folder(folder_name,limit_month=None)

from floris.tools import FlorisInterface
from utilities.helpers import adaptive_timeit
settings_path = Path("utilities") / "floris_settings.yaml"
fi = FlorisInterface(settings_path)
fi.reinitialize()

aep = fi.get_farm_AEP_wind_rose_class(
    wind_rose=fl_wr,
)

a = aep / (365 * 24* 10**6)
print(a)

#%%

""" 
calculates the aep of a wind farm subject to directions theta_i Settings are taken from the "floris_settings.yaml". 
The execution time is measured by the adaptive_timeit function (5 repeats over ~1.5 seconds) - this shouldn't effect the aep result.

Args:
    fl_wr: floris wind rose object
    theta_i (bins,): In radians! Angle of bin (North is 0, clockwise +ve -to match compass bearings)
    layout (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines. Normalised by rotor diameter!
    turb (turbine obj) : turbine object, must have turb.D attribute to unnormalise layout
    wake (boolean): set False to run fi.calculate_no_wake()
    timed (boolean) : set False to run without timings (timing takes 4-8sec)

Returns:
    pow_j (nt,) : aep of induvidual turbines
    time : lowest execution time measured by adaptive timeit 
"""
wake = True
timed= False
from pathlib import Path
from floris.tools import FlorisInterface
from utilities.helpers import adaptive_timeit
settings_path = Path("utilities") / "floris_settings.yaml"
fi = FlorisInterface(settings_path)

wd_array = np.array(fl_wr.df["wd"].unique(), dtype=float)
if len(thetaD_i) != len(wd_array):
    raise ValueError("Floris is using a different amount of bins to FLOWERS (?!): len(thetaD_i) != len(wd_array)")
ws_array = np.array(fl_wr.df["ws"].unique(), dtype=float)
wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
from scipy.interpolate import NearestNDInterpolator
freq_interp = NearestNDInterpolator(fl_wr.df[["wd", "ws"]],fl_wr.df["freq_val"])
freq = freq_interp(wd_grid, ws_grid)
freq_2D = freq / np.sum(freq)

turb_type = [turb.name,]
fi.reinitialize(wind_directions=wd_array,wind_speeds=ws_array,time_series=False)

fi.reinitialize(layout_x=turb.D*layout[:,0], layout_y=turb.D*layout[:,1])

if wake:
    _,time = adaptive_timeit(fi.calculate_wake,timed=timed)
else:
    _,time = adaptive_timeit(fi.calculate_no_wake,timed=timed)

pow_j = np.nansum(fi.get_turbine_powers()*freq_2D[...,None],axis=(0,1)) 

print(pow_j/(1*10**6))



#%%
a,b = floris_FULL_timed_aep(fl_wr,thetaD_i,layout,turb,timed=False)
print(a)
#%%
wd_array = np.array(fl_wr.df["wd"].unique(), dtype=float)
ws_array = np.array(fl_wr.df["ws"].unique(), dtype=float)
wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
freq_interp = NearestNDInterpolator(fl_wr.df[["wd", "ws"]],fl_wr.df["freq_val"])
freq = freq_interp(wd_grid, ws_grid)
freq_2D = freq / np.sum(freq)

fi.reinitialize(layout_x=turb.D*layout[:,0], layout_y=turb.D*layout[:,1],wind_directions=wd_array,wind_speeds=ws_array,time_series=False)
fi.calculate_wake()
c = np.sum(fi.get_farm_power() * freq_2D * 8760)
#%%
d = np.sum(fi.get_turbine_powers()*freq_2D[...,None],axis=(0,1)) 

#%%
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')
wd_array = np.array(fl_wr.df["wd"].unique(), dtype=float)
ws_array = np.array(fl_wr.df["ws"].unique(), dtype=float)
wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
freq_interp = NearestNDInterpolator(fl_wr.df[["wd", "ws"]],fl_wr.df["freq_val"])
freq = freq_interp(wd_grid, ws_grid)

ax.imshow(freq)

#%%

if wake:
    _,time = adaptive_timeit(fi.calculate_wake,timed=timed)
else:
    _,time = adaptive_timeit(fi.calculate_no_wake,timed=timed)

aep_array = fi.get_turbine_powers()
pow_j = np.sum(P_i[:, None, None]*aep_array, axis=0)  # weight average using probability


#%%Familiarising myself with the (jensen) FLOWERS libary ..








































import flowers.model_interface as inter
import flowers.tools as tl
import numpy as np
import matplotlib.pyplot as plt

wr = tl.load_wind_rose(7)
layout_x = 126 * np.array([0.])
layout_y = 126 * np.array([0.])
model = inter.AEPInterface(wr, layout_x, layout_y)
model.reinitialize(ws_avg=True)
model.compare_aep()

#%%
%load_ext autoreload
%autoreload 2

import numpy as np
from pathlib import Path
from floris.tools import WindRose
import flowers.model_interface as inter
import flowers.tools as tl

from utilities.turbines import iea_10MW
turb = iea_10MW()

layout = np.array([[0,0],[5,0]]) *turb.D

current_file_path = Path(__file__)
folder_name = current_file_path.parent.parent/ "data" / "WindRoseData_D" / ("site"+str(6)) #location of Wind toolkit csv download
floris_wr = WindRose() # floris wind rose object
#parse the csv and bin into 5 deg bins (default)
floris_wr.parse_wind_toolkit_folder(folder_name,limit_month=None) 
# so just pass this to the FLOWERS library?
wr = floris_wr.df #just the dataframe


import flowers.flowers_interface as flow
flower_int = flow.FlowersInterface(wr, turb, layout, num_terms=37, k=0.05) #nrel_5MW iea_10MW
aep = flower_int.calculate_aep()

print(aep)
#%%
import numpy as np

import flowers.model_interface as inter
import flowers.tools as tl


layout = np.array([[0,0],[5,0]])*turb.D

from utilities.helpers import get_floris_wind_rose
U_i,P_i = get_floris_wind_rose(6)
thetaD_i = np.linspace(0,360,72,endpoint=False)

import flowers.flowers_interface as flow
flower_int = flow.FlowersInterface(U_i,P_i,thetaD_i, layout, turb,num_terms=37, k=0.05)
aep = flower_int.calculate_aep()
print(aep)
#%%

from utilities.helpers import get_floris_wind_rose
U_i,P_i = get_floris_wind_rose(6)
thetaD_i = np.linspace(0,360,72,endpoint=False)

a = wr.copy(deep=True)
a = tl.resample_average_ws_by_wd(a)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.scatter(np.arange(0,len(P_i),1),a.freq_val.values/np.sum(a.freq_val.values),marker='x')
ax.scatter(np.arange(0,len(U_i),1),P_i,marker='+')


#%%
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')
ax.scatter(np.arange(0,len(U_i),1),a.ws.values,marker='x')
ax.scatter(np.arange(0,len(U_i),1),U_i,marker='+')

