#%% Find the optimal rotation of the square layouts
#%% get data to plot
import sys
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2

import numpy as np

run = 1
SAVE_FIG = False
timed = True 

U_LIM = 3 #manually override ("user limit") the invalid radius around the turbine (otherwise variable, depending on k/Ct) - 
RESOLUTION = 100 #number of x/y points in contourf meshgrid
EXTENT = 30 #total size of contourf "window" (square from -EXTENT,-EXTENT to EXTENT,EXTENT)
K = 0.03 #expansion parameter for the Gaussian model
Kj = 0.04 #expansion parameter for the Jensen model
NO_BINS = 72 #number of bins in the wind rose

site_n = 5
from utilities.turbines import iea_10MW
turb = iea_10MW()

thetaD_i = np.linspace(0,360,NO_BINS,endpoint=False) #theta in degrees
thetaD_WB_i = 270 - thetaD_i #wind bearing bin centers 

from utilities.helpers import get_floris_wind_rose,rectangular_layout,vonMises_wr
# U_i,P_i,_,fl_wr = get_floris_wind_rose(site_n,align_west=True)
U_AV = 10
kappa = 5

U_i,P_i,_ = vonMises_wr(U_AV,kappa)

turb_n = 9
spacing = 7

#U_avs     : kappa    : turb_n: spacing : rots
#[ 5,10,15]:[ 5, 5, 5]:[7,7,7]:[7,7,7]:[45,45,45]
#[10,10,10]:[ 1, 5,20]:[7,7,7]:[7,7,7]:[45,45,25]
#[10,10,10]:[ 5, 5, 5]:[5,7,9]:[7,7,7]:[45,45,45]
#[10,10,10]:[ 5, 5, 5]:[7,7,7]:[5,7,9]:[45,45,45]

max_r = 45
rots = list(range(0, max_r + 1, 5))
pow_j = np.zeros((len(rots),turb_n**2))

from utilities.AEP3_functions import floris_AV_timed_aep

for i in range(len(rots)):
    layout = rectangular_layout(turb_n,spacing,np.deg2rad(rots[i]))
    pow_j[i,:],_ = floris_AV_timed_aep(U_i,P_i,thetaD_WB_i,layout,turb,timed=timed) 
    if i==0 or i==len(rots)-1 or i % 10 == 0:
        print(f"{i+1}/{len(rots)}")

aep = np.sum(pow_j,axis=1)
idx = np.argmax(aep)

print(f"U_AV:{U_AV},kappa:{kappa},turb_n:{turb_n},spacing:{spacing}, rot:{rots[idx]}, max:{aep[idx]:.1f}")
      
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(3,0.3),dpi=200)
ax.scatter(rots,aep,marker='x')


#site_n : rots : turb_n
# [2,1,8]:[30,40,25]:[7,7,7]
#
#for U_AV = 10, in kappa : rotation format
# 1 : 45
# 5 : 45
# 20: 25

