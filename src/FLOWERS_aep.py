#%% setup cell
import numpy as np
import sys
import os
sys.path.append(os.path.join('..', 'src')) #allow to import from utilities (there may be a better way ...)
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2

from utilities.turbines import iea_10MW
turb = iea_10MW()

from utilities.helpers import get_floris_wind_rose
U_i,P_i,a = get_floris_wind_rose(6)
thetaD_i = np.linspace(0,360,72,endpoint=False)
layout = np.array([[0,0]])

from utilities.AEP3_functions import flowers_timed_aep
from utilities.helpers import adaptive_timeit
from pathlib import Path

pow_z,time_1 = flowers_timed_aep(U_i,P_i,thetaD_i,layout,turb,0.05,timed=True)
aep1 = np.sum(pow_z) 

from utilities.helpers import simple_Fourier_coeffs,get_WAV_pp
_,Fourier_coeffs3_PA = simple_Fourier_coeffs(turb.Cp_f(U_i)*(P_i*(U_i**3)*len(P_i))/(2*np.pi))
wav_Ct = get_WAV_pp(U_i,P_i,turb,turb.Ct_f)
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

from utilities.plotting_funcs import si_fm
print(f'JFLOWERS: {aep1:.2f} in {si_fm(time_1)}')
print(f'GFLOWERS: {aep2:.2f} in {si_fm(time_2)}')

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

