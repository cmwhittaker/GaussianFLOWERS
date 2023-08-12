#%% Figure to show the wind roses and the original speed and frequency distributions used in the evaluation of the Ctag vs Floris vs CubeAv Evaluation
#get the wind rose data

%load_ext autoreload
%autoreload 2

import numpy as np
from floris.tools import WindRose
def get_floris_wind_rose(site_n):
    fl_wr = WindRose()
    folder_name = "WindRoseData_D/site" +str(site_n)
    fl_wr.parse_wind_toolkit_folder(folder_name,limit_month=None)
    wr = fl_wr.resample_average_ws_by_wd(fl_wr.df)
    wr.freq_val = wr.freq_val/np.sum(wr.freq_val)
    U_i = wr.ws
    P_i = wr.freq_val
    return U_i,P_i

from turbines_v01 import iea_10MW
turb = iea_10MW()

NO_BINS = 72
site_n = [2,3,6]
U_i,P_i = [np.zeros((NO_BINS,len(site_n))) for _ in range(2)]
theta_i = np.linspace(0,2*np.pi,NO_BINS)

for i in range(len(site_n)):
    U_i[:,i],P_i[:,i] = get_floris_wind_rose(site_n[i])

#%% plot that wind rose data
from AEP3_plot_funcs_v01 import nice_polar_plot_v01
SAVE_FIG = True

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 5, wspace=0.1,hspace=0.1)
fig = plt.figure(figsize=(7.8,8), dpi=300)
rlp = 0

frac1 = 1.05
lim5 = None #np.max(P_i*U_i)
lim1 = None #np.max(P_i)
lim2 = np.max(U_i)*frac1
lim3 = np.max(turb.Ct_f(U_i))*frac1
lim4 = np.max(turb.Cp_f(U_i))*frac1

for i in range(len(site_n)):
    nice_polar_plot_v01(fig,gs[i,0],theta_i,U_i[:,i]*P_i[:,i],"$P(\\theta)U(\\theta)$",ylim=[None,lim5],rlp=rlp)
    nice_polar_plot_v01(fig,gs[i,1],theta_i,P_i[:,i],"$P(\\theta$",bar=False,ylim=[None,lim1],rlp=rlp)
    nice_polar_plot_v01(fig,gs[i,2],theta_i,U_i[:,i],"$U(\\theta)$",bar=False,ylim=[None,lim2],rlp=rlp)
    nice_polar_plot_v01(fig,gs[i,3],theta_i,turb.Ct_f(U_i[:,i]),"$C_t(U(\\theta))$",bar=False,ylim=[None,lim3],rlp=rlp)
    nice_polar_plot_v01(fig,gs[i,4],theta_i,turb.Cp_f(U_i[:,i]),"$C_p(U(\\theta))$",bar=False,ylim=[None,lim4],rlp=rlp)

sitestr = ""
for _ in site_n:
    sitestr =sitestr+ str(_) 

if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+"_"+sitestr+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')
    print("figure saved")
        
#%%
for _ in site_n:
    print(str(_))