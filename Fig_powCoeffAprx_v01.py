#% A figure to show the effect of setting the POWER coefficient as a constant based on the mean weight-averaged velocity.
#(A 3 x 3 pannel of the wake velocity deficit error for a WIND ROSE input)
#%% get data to plot
import numpy as np

def rectangular_domain(extent,r=200):
    X,Y = np.meshgrid(np.linspace(-extent,extent,r),np.linspace(-extent,extent,r))
    return X,Y,np.column_stack((X.reshape(-1),Y.reshape(-1)))

K = 0.03
INVALID_R = 5
SAVE_FIG = False
site_list=[2,3,6]
no_bins = 72
wr_list = []
fixed_Cp_list = []
Z_list = []
from distributions_vC05 import wind_rose
from AEP3_2_functions import y_5MW
turb = y_5MW()
from AEP3_3_functions import cubeAv_v4,gen_local_grid_v01C
#get domain
layout = np.array(((0,0),))
X,Y,plot_points = rectangular_domain(15,r=200)

def pce(actual,approx):
    return 100*((actual-approx)/actual)

for i in range(len(site_list)):
    #first and second columns are wind rose related
    wr = wind_rose(bin_no_bins=no_bins,custom=None,site=site_list[i],filepath=None,a_0=1,Cp_f=turb.Cp_f)
    wr_list.append(wr)
    #third column is the contour function
    r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
    Cp_f = np.sum(turb.Cp_f(wr.avMagnitude)*wr.frequency) #fixed
    fixed_Cp_list.append(Cp_f)
    Ct_f = turb.Ct_f
    actual_Z,_,_  = cubeAv_v4(r_jk,theta_jk,
                       np.linspace(0,2*np.pi,wr.bin_no_bins,endpoint=False),
                       wr.avMagnitude,
                       wr.frequency,
                       Ct_f, #Thrust coefficient function
                       Cp_f,
                       K,
                       turb.A)  
    approx_Z,_,_ = cubeAv_v4(r_jk,theta_jk,
                    np.linspace(0,2*np.pi,wr.bin_no_bins,endpoint=False),
                    wr.avMagnitude,
                    wr.frequency,
                    Ct_f, #fixed Ct
                    Cp_f,
                    K,
                    turb.A)
    Z = pce(actual_Z,approx_Z).reshape(X.shape)
    Z = np.where(np.sqrt(X**2+Y**2)<INVALID_R,np.nan,Z)
    Z_list.append(Z)   
    
#%% Set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

def nice_polar_plot(fig,gs,x,y,text):
    ax = fig.add_subplot(gs,projection='polar')
    ax.plot(x,y,color='black',linewidth=1)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(50)  # Move radial labels away from plotted line
    from matplotlib.ticker import MaxNLocator
    #ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.text(0, 0, text, ha='left',transform=ax.transAxes,color='black')
    ax.spines['polar'].set_visible(False)
    return None

def nice_cartesian_plot(fig,gs,x,y,row,ybar):
    axs_off = [0,0.2,0.2]
    text_off = [-0.0005,-0.0005,-0.02] #[0.0005,0.0005,0.02]
    ax1 = fig.add_subplot(gs)
    ax1.plot(x,y,color='black',linewidth=1)
    upper = np.max(y) + axs_off[row]*(np.max(y)-np.min(y))
    ax1.set_ylim(np.min(y),upper)
    ax1.hlines(y=ybar,xmin=0,xmax=360,linewidth=1,color='blue',ls='--')
    annotation_txt = "$\overline{C_t}=" + f'{ybar:.2f}$'  
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0)  
    ax1.annotate(annotation_txt, xy=(360,ybar+text_off[row]), ha='right', va='top',color='blue',bbox=props)
    
    ax1.set_xlabel('$\\theta$ / $^\circ$',labelpad=0)
    ax1.set_ylabel('$C_t(\\theta)$',labelpad=2) 

    # ax2 = ax1.twinx()
    # ax2.plot(x,wr.frequency)
    # ax2.set_ylabel('',labelpad=2)
    return None

from matplotlib import cm
def nice_contour_plot_v02(fig,gs,row,X,Y,Z,xt,yt):
    ax = fig.add_subplot(gs[row,2])
    cf = ax.contourf(X,Y,Z,cmap=cm.coolwarm)
    xticks = ax.xaxis.get_major_ticks()
    xticks[1].set_visible(False)
    ax.set_xlabel('$x/d_0$',labelpad=-9)

    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-19)

        #colourbar and decorations
    cax = fig.add_subplot(gs[row+1,2])
    if row != 0:
        cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.2g')
    else:
        cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.1g')
    cb.ax.locator_params(nbins=6)
    cb.ax.tick_params(labelsize=7)

    if row == 4:
        cax.set_xlabel('Percentage Error in AEP / \%',labelpad=2)
        
    ax.scatter(xt,yt,marker='x',color='black',s=10)

    return None

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(6, 3, height_ratios=[14,1,14,1,14,1],wspace=0.25,hspace=0.37)
fig = plt.figure(figsize=(6.7,7), dpi=200)

import numpy as np

for i in range(len(site_list)): #for each row
    #first column is the wind rose
    wr = wr_list[i]
    wind_rose_y = wr.frequency*wr.avMagnitude
    nice_polar_plot(fig,gs[2*i,0],np.deg2rad(wr.deg_bin_centers),wind_rose_y,text="$P(\\theta)U(\\theta)$")
    #second column is the thrust coefficient curve

    nice_cartesian_plot(fig,gs[2*i,1],wr.deg_bin_centers,turb.Cp_f(wr.avMagnitude),i,fixed_Cp_list[i])
    #third column is the percentage error contour plot
    nice_contour_plot_v02(fig,gs,2*i,X,Y,Z_list[i],0,0)

if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches="tight")

#%% 
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
xs = np.linspace(0,360,wr.bin_no_bins)
ax.plot(wr.avMagnitude)
ax2 = ax.twinx()
ax2.plot(wr.frequency,color='orange')
ax3 = ax.twinx()
ax3.plot(turb.Ct_f(wr.avMagnitude),color='green')
ax3.axhline(turb.Ct_f(np.mean(wr.avMagnitude*wr.frequency*72)),color='green')
ax4 = ax.twinx()
#%%
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
ax.plot(wr.avMagnitude*wr.frequency*72,color='black')
ax.axhline(np.sum(wr.avMagnitude*wr.frequency),color='black')