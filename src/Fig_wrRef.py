#%% Figure to show the wind roses, their constiuents (P_i and U_i) and the variation in thrust and power coefficient with direction (C_t(U_i and C_p(U_i)))

#get the wind rose data
%load_ext autoreload
%autoreload 2

import numpy as np
from utilities.turbines import iea_10MW
turb = iea_10MW()

NO_BINS = 72
SAVE_FIG = False
site_n = [2,1,8]
site_n = list(range(1,12))
U_i,P_i = [np.zeros((NO_BINS,len(site_n))) for _ in range(2)]
theta_i = np.linspace(0,2*np.pi,NO_BINS,endpoint=False)

from utilities.helpers import get_floris_wind_rose
for i in range(len(site_n)):
    U_i[:,i],P_i[:,i],a,_ = get_floris_wind_rose(site_n[i],align_west=True)

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)
#%% plot data

def nice_polar_plot(fig,gs,x,y,ann_txt,bar=True,ylim=None,rlp=0):
    ax = fig.add_subplot(gs,projection='polar')
    if bar:
        ax.bar(x,y,color='grey',linewidth=1,width=2*np.pi/72)
    else:
        ax.plot(x,y,color='black',linewidth=1)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(rlp)  # Move radial labels away from plotted line
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)
    ax.annotate(ann_txt, xy=(0.4,0.75), ha='center', va='bottom',color='black',xycoords='axes fraction',rotation='vertical',bbox=props)
    ax.spines['polar'].set_visible(False)
    ax.set(ylim=ylim)
    return None

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(len(site_n), 5, wspace=0.1,hspace=0.1)
fig = plt.figure(figsize=(7.8,2*len(site_n)), dpi=300)
rlp = 0

frac1 = 1.05
lim5 = None #np.max(P_i*U_i)
lim1 = None #np.max(P_i)
lim2 = np.max(U_i)*frac1
lim3 = np.max(turb.Ct_f(U_i))*frac1
lim4 = np.max(turb.Cp_f(U_i))*frac1

theta_WB_i = (3*np.pi/2) - theta_i

for i in range(len(site_n)):
    
    nice_polar_plot(fig,gs[i,0],theta_WB_i,U_i[:,i]*P_i[:,i],"$P(\\theta)U(\\theta)$",ylim=[None,lim5],rlp=rlp)
    nice_polar_plot(fig,gs[i,1],theta_WB_i,P_i[:,i],"$P(\\theta$)",bar=False,ylim=[None,lim1],rlp=rlp)
    nice_polar_plot(fig,gs[i,2],theta_WB_i,U_i[:,i],"$U(\\theta)$",bar=False,ylim=[None,lim2],rlp=rlp)
    nice_polar_plot(fig,gs[i,3],theta_WB_i,turb.Ct_f(U_i[:,i]),"$C_t(U(\\theta))$",bar=False,ylim=[None,lim3],rlp=rlp)
    nice_polar_plot(fig,gs[i,4],theta_WB_i,turb.Cp_f(U_i[:,i]),"$C_p(U(\\theta))$",bar=False,ylim=[None,lim4],rlp=rlp)

if SAVE_FIG:
    from pathlib import Path

    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "fig images"
    fig_name = f"Fig_wrRef.png"
    image_path = fig_dir / fig_name

    plt.savefig(image_path, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")


#%% just examine a certain one
#[-70,210,-35,250,250,270,190,-35,215,245,265,145]
a,b,c,_ = get_floris_wind_rose(11,align_west=False)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(dpi=200)
ax.plot(np.rad2deg(theta_WB_i),a*b)
ax.vlines(265,-1,5,color='red')
ax.set(ylim=(0,np.max(a*b)))


#%%



mean_ws = np.sum(U_i*P_i,axis=0)