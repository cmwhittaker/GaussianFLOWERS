#%% validating the "wrapper" I've used around floris
# flow field first
# unfortunately, this only works for a single wind direction, and the plot is aligned with the wind direction (not the layout coordinates)
import numpy as np
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

#from utilities.helpers import get_floris_wind_rose_WB
#U_i,P_i,theta_i,_ = get_floris_wind_rose_WB(6)

theta_dash = 360
U_i,P_i,theta_WB_i = np.array((10,)),np.array((1,)),np.array((np.deg2rad(270-theta_dash),)) 

from pathlib import Path
from floris.tools import FlorisInterface

def rotate_layout(layout,rot):
    #rotates layout anticlockwise by angle rot (in radians)
    Xt,Yt = layout[:,0],layout[:,1]
    rot_Xt = Xt * np.cos(rot) - Yt * np.sin(rot)
    rot_Yt = Xt * np.sin(rot) + Yt * np.cos(rot) 
    layout_r = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
    return layout_r

layout = np.array(((-3,0),(0,0),(-0.2,-3),(-0.4,-6)))

settings_path = Path(__file__).parent.parent/ "src" / "utilities" / "floris_settings.yaml"
fi = FlorisInterface(settings_path)
fi.reinitialize(wind_directions=np.rad2deg(theta_WB_i), wind_speeds=U_i, time_series=True, layout_x=turb.D*layout[:,0], layout_y=turb.D*layout[:,1])

horizontal_plane = fi.calculate_horizontal_plane(x_resolution=200, y_resolution=200, height=119.0)

from floris.tools.visualization import visualize_cut_plane
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
visualize_cut_plane(horizontal_plane, ax=ax, title="Horizontal")
ax.scatter(turb.D*layout[:,0],turb.D*layout[:,1],marker='x',color='black')

#%%
cut_plane = horizontal_plane
x1_mesh = cut_plane.df.x1.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])
x2_mesh = cut_plane.df.x2.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])

vel_mesh = cut_plane.df.u.values.reshape(cut_plane.resolution[1], cut_plane.resolution[0])

mesh_stack = np.column_stack((x1_mesh.reshape(-1),x2_mesh.reshape(-1)))
mesh_stack_r = rotate_layout(mesh_stack,np.deg2rad(theta_dash))
x1_mesh_r = mesh_stack_r[:,0].reshape(x1_mesh.shape)
x2_mesh_r = mesh_stack_r[:,1].reshape(x2_mesh.shape)

import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
cf = ax.pcolormesh(x1_mesh_r,x2_mesh_r,vel_mesh,cmap=cm.coolwarm)
fig.colorbar(cf)
ax.set_aspect("equal")