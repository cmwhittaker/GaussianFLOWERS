#%% general testing script

from src.utilities.turbines import iea_15MW
turb = iea_15MW()

#%% 
import numpy as np
from src.utilities.helpers import rectangular_layout
layout = rectangular_layout(3,4,np.deg2rad(10))
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')
ax.scatter(layout[:,0],layout[:,1])