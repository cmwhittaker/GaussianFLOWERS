#%% Plotting the CT and CP curves used for the report
from soa_functions_vC01 import VestaV80

turbine = VestaV80()
import numpy as np
u = np.linspace(0,30,300,endpoint=True)

Ct = turbine.C_t(u)
Cp = turbine.C_p(u)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(4,4),dpi=200)
l1 = ax.plot(u,Ct,linestyle='dashed',linewidth=1,label='$C_t$')
l2 = ax.plot(u,Cp,linestyle='dotted',linewidth=1,label='$C_p$')
ax.set_xlabel('Hub Velocity / $ms^{-1}$')

ax.legend()
#%%
bbox = fig.bbox_inches.from_bounds(0,0,3.65,3.55) #Crop
plt.savefig(r"AEP3_Evaluation_Report_v01\Figures\CtCpCurve.png",dpi='figure',format='png',bbox_inches=bbox)


#%% set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)