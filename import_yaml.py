#%% Script to import the turbine data into a numpy array

import numpy as np
import yaml

# Load the YAML data

filepath = r'C:\Users\Work\anaconda3\envs\Windy4\Lib\site-packages\floris\turbine_library\x_20MW.yaml'
with open(filepath, 'r') as file:
    data = yaml.safe_load(file)

power_thrust_table = data['power_thrust_table']

# Convert each list to a numpy array
power = np.array(power_thrust_table['power'])
wind_speed = np.array(power_thrust_table['wind_speed'])
thrust = np.array(power_thrust_table['thrust'])

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(6,4),dpi=400)
ax.plot(wind_speed,power,label='Cp',linewidth=1)
ax.plot(wind_speed,thrust,label='Ct',linewidth=1)
ax.legend()
#%%
bbox = fig.bbox_inches.from_bounds(0,0,6,4) #Crop
plt.savefig(r'AEP3_Evaluation_Report_v02\Figures\CtCpCurve_y5MW.png',dpi='figure',format='png',bbox_inches=bbox)

#%%
Cp = np.interp(8,wind_speed,power)
print(Cp)

#%%
from AEP3_functions_v01 import y_5MW
turbine = y_5MW()

plt.plot(turbine.wind_speed,turbine.Ct,label='Ct',color='red')
plt.plot(turbine.wind_speed,turbine.Cp,label='Cp')
ax.legend()