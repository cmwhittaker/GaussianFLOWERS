#%% this is the simple appendix figure showing the negative power predicion

import numpy as np
from utilities.turbines import iea_10MW
turb = iea_10MW()
K=0.03 #wake expansion rate

SAVE_FIG = False
NT = 4 #number of turbines 
SPACING = 5 #turbine spacing normalised by rotor DIAMETER
XPAD = 7 #X border in normalised rotor DIAMETERS
YPAD = 7 #Y border in normalised rotor DIAMETERS
U_inf = 10 #inflow wind speed
U_LIM = None #manually override ("user limit") the invalid radius around the turbine (otherwise variable, depending on k/Ct) - 
K = 0.03
XRES = 400 #number of x points in the contourf meshgrid
YRES = 200 #must be odd so centerline can be picked later on

from utilities.helpers import linear_layout,rectangular_domain
xt,yt,layout = linear_layout(NT,SPACING)
xx,yy,plot_points,_,_ = rectangular_domain(layout,xpad=XPAD,ypad=YPAD,xr=XRES,yr=YRES)

td = 0
U_i,P_i,thetaD_i = np.array((U_inf,)),np.array((1,)),np.array((td,)) 

from utilities.AEP3_functions import num_Fs

#flow field (for visualisation - no timing) using num_F
powj_a,Uwt_a,Uwff_a = num_Fs(U_i,P_i,np.deg2rad(thetaD_i),
                            plot_points,layout,
                            turb,K,
                            u_lim=None,
                            Ct_op=1, 
                            Cp_op=1, 
                            cross_ts=True,ex=True, 
                            ff=True)
aep_a = np.sum(powj_a)

powj_b,Uwt_b,Uwff_b = num_Fs(U_i,P_i,np.deg2rad(thetaD_i),
                            plot_points,layout,
                            turb,K,
                            u_lim=None,
                            Ct_op=1, 
                            Cp_op=1, 
                            cross_ts=False,ex=True, #NO CROSS TERMS
                            ff=True,cube_term=False)
aep_b = np.sum(powj_b)

#% package results into a "nice" figure

import matplotlib.pyplot as plt
from utilities.plotting_funcs import set_latex_font
set_latex_font()
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 1, height_ratios=[14,1,1,10],wspace=0.2,hspace=0.2)
fig = plt.figure(figsize=(7.8,5), dpi=300) 
ax1 = fig.add_subplot(gs[0,0])
cax = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[3,0])

xlims = (-1,np.max(layout[:,0])+3)
a1 = 2
ylims = (-a1,a1)

#contourf first
ax1.set_xlabel('$x/d_0$',labelpad=-7)
ax1.set_ylabel('$y/d_0$',labelpad=-12)
yticks = ax1.yaxis.get_major_ticks()
yticks[1].set_visible(False)
ax1.set(aspect='equal',xlim=xlims,ylim=ylims)
ax1.set_aspect('equal')
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
colors = ["black", "white"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
cf = ax1.contourf(xx,yy,Uwff_b.reshape(xx.shape),50,cmap=cmap1)
ax1.scatter(layout[:,0],layout[:,1],marker ='x',color='black')
#mark turbine locations
for i in range(NT): #label each turbine
    an_text = str(i+1)
    ax1.annotate(an_text, xy=(xt[i],yt[i]-.3), ha='center', va='top',color='black',fontsize=6)

#colour bar
cb = fig.colorbar(cf, cax=cax, cmap=cmap1,orientation='horizontal',format='%.3g')
cb.set_label("Wind Velocity / $ms^{-1}$")
from matplotlib.ticker import MaxNLocator
cax.xaxis.set_major_locator(MaxNLocator(integer=True))

#scatter of per-turbine power generation
ax2.set_xlabel('Turbine No.')
ax2.set_ylabel('Power generation / MW')
ax2.set(xlim=xlims)
ax2.plot(layout[:,0],powj_a,marker = 'x',color='black',label='Before simplification')
ax2.plot(layout[:,0],powj_b,marker='+',color='grey',label='After simplifcation')
ax2.hlines(0,-20,20,ls='--',color='grey',lw=1)
ax2.legend(loc='upper right')
xtick_labels = [str(i+1) for i in range(0,NT)]
ax2.set_xticks(layout[:,0])
ax2.set_xticklabels(xtick_labels)

if SAVE_FIG: 

    from pathlib import Path
    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "fig images"
    fig_name = f"Fig_XtermSingle_v2_{SPACING}D_{NT}T_{U_inf}ms_{td}deg.png"
    path_plus_name = fig_dir / fig_name
    
    plt.savefig(path_plus_name, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")
    print(f"to {path_plus_name}")

#%% 
xs = np.linspace(0,30,100)
ys = turb.Ct_f(xs)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.plot(xs,ys)