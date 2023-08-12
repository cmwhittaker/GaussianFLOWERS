#%% Figure to show the wind roses and the original speed and frequency distributions used in the evaluation of the Ctag vs Floris vs CubeAv Evaluation

#sites = [f+1 for f in range(12)]
sites = [1,2,3,4,5,6,7,8,9,10,11]
no_sites = len(sites)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
gs = GridSpec(no_sites, 3,wspace=0.3,hspace=0.37)
fig = plt.figure(figsize=(6,2*no_sites), dpi=200)

from distributions_vC05 import wind_rose
import numpy as np
bin_no_bins = 72
theta = np.linspace(0,2*np.pi,bin_no_bins)

def nice_wind_rose(fig,gs,row,column,x,y,text,text2=None):
    #first column is the wind rose
    ax = fig.add_subplot(gs[row,column],projection='polar')
    ax.plot(x,y,color='black')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(60)  # Move radial labels away from plotted line
    ax.text(0, 0.9, text, ha='left',transform=ax.transAxes,color='coral')

    if column == 0 :
        text2 = "site:{}/min:{:.1f}/max:{:.1f}/mean:{:.1f}\n".format(sites[row],np.min(y),np.max(y),np.mean(y))
        ax.text(0, 0, text2, ha='left',transform=ax.transAxes,color='coral',fontsize=6)

    if column == 2:
        text2 = "min:{:.1f}/max:{:.1f}/sum:{:.1f}\n".format(np.min(y),np.max(y),np.sum(y))
        ax.text(0, 0, text2, ha='left',transform=ax.transAxes,color='coral',fontsize=6)
    return None

for i in range(no_sites): #there are 12 subplots
    wr = wind_rose(bin_no_bins=bin_no_bins,site=sites[i])
    nice_wind_rose(fig,gs,i,0,theta,wr.avMagnitude,'$U[\\theta]$',True) #
    nice_wind_rose(fig,gs,i,1,theta,wr.frequency,'$P[\\theta]$')
    nice_wind_rose(fig,gs,i,2,theta,wr.djd,'$P[\\theta]U[\\theta]$')

bbox = fig.bbox_inches.from_bounds(0.5,0.5,5,5) #Crop
plt.savefig(r"AEP3_Evaluation_Report_v02\Figures\WindRosePlot_v02.png",dpi='figure',format='png',bbox_inches=bbox)


#%% set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

