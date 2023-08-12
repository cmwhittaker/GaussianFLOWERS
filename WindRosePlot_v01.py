#%% A plot of the wind roses I'm using for AEP estimates

from depreciated.distributions_vC03 import wind_rose

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
#create figure 
gs = GridSpec(4, 3,wspace=0,hspace=0.1)
fig = plt.figure(figsize=(6,6), dpi=200)

import numpy as np
bin_no_bins = 72
theta = np.linspace(0,2*np.pi,bin_no_bins)

for i in range(12): #there are 12 subplots
    wr = wind_rose(bin_no_bins=bin_no_bins,site=i+1)

    ax = fig.add_subplot(gs[i],projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['', '45', '', '135', '', '225', '', '315'])
    ax.plot(theta,wr.og_djd)
    ax.text(0.5, 0.9, 'site ' + str(i+1), ha='center',transform=ax.transAxes,color='coral')

bbox = fig.bbox_inches.from_bounds(0.5,0.5,5,5) #Crop
plt.savefig(r"AEP3_Evaluation_Report_v01\Figures\WindRosePlot_v01.png",dpi='figure',format='png',bbox_inches=bbox)

#%% set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)