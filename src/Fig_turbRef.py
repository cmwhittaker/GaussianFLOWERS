#%% Figure to show the power/thrust coefficient curves 

import numpy as np
from utilities.turbines import iea_10MW
from utilities.plotting_funcs import set_latex_font
set_latex_font()
turb = iea_10MW()
xs = np.linspace(0,30,100)
y1 = turb.Ct_f(xs)
y2 = turb.Cp_f(xs)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(3,2.5),dpi=300)
ax.plot(xs,y1,label='$C_t$')
ax.plot(xs,y2,label='$C_p$')
ax.set(xlabel='Wind Speed / $ms^{-1}$',ylabel = 'Thrust/Power Coefficient',ylim=(None,0.9))

ax.legend(loc='upper right')

# Adding grey boxes to highlight regions
ax.axvspan(3, 10.5, color='grey', alpha=0.15,ec='none')  # Adjust the x-axis values as per the region you want to highlight
ax.axvspan(11, 25, color='grey', alpha=0.15,ec='none')  # Adjust the x-axis values as per the region you want to highlight

ax.text(6.75, 0.85, 'Region II', horizontalalignment='center', verticalalignment='center', color='black')
ax.text(18.25, 0.85, 'Region III', horizontalalignment='center', verticalalignment='center', color='black')

ax.vlines(7,0,10)
