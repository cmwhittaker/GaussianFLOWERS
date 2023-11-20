#%% this is a simple, single direction case study 

#%% set font!
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

#%%generate the data
import numpy as np
SAVE_FIG = False

NT = 5 #number of turbines 
SPACING = 7 #turbine spacing normalised by rotor DIAMETER
XPAD = 7 #X border in normalised rotor DIAMETERS
YPAD = 7 #Y border in normalised rotor DIAMETERS
BINS = 360 #number of wind bins
U_inf = 10 #inflow wind speed
U_LIM = 4 #manually override ("user limit") the invalid radius around the turbine (otherwise variable, depending on k/Ct) - 
K = 0.03

CP_LOWER = 0 #wind speed lower lim for plotting
CP_UPPER = 27 #wind speed upper lim for plotting

EXTENT = 14 #extent of rectangular domain

XRES = 300 #number of x points in the contourf meshgrid
YRES = 101 #must be odd so centerline can be picked later on

def impulse_wr(bin):
    #single direction ALIGNED
    U_i = np.ones(720,)*U_inf
    P_i = np.zeros_like(U_i)
    P_i[bin] = 1 #blows from a single wind direction
    theta_i = np.linspace(0,2*np.pi,720,endpoint=False)
    return U_i,P_i,theta_i

from utilities.turbines import iea_10MW
turb = iea_10MW()
Ct_f = turb.Ct_f
Cp_f = turb.Cp_f

ALIGNED = True #pick between algined and not
if ALIGNED:
    non_zero_bin = 540
else:
    non_zero_bin = 545
U_i,P_i,theta_i = impulse_wr(non_zero_bin)

from utilities.helpers import linear_layout,rectangular_domain
xt,yt,layout = linear_layout(NT,SPACING)
xx,yy,plot_points,_,_ = rectangular_domain(layout,xpad=XPAD,ypad=YPAD,xr=XRES,yr=YRES)

from utilities.AEP3_functions import num_Fs
#firstly with the cross terms
powj_a,_,Uwff_a= num_Fs(U_i,P_i,theta_i,
                        layout,
                        plot_points,
                        turb,
                        K=K,
                        RHO=1.225,
                        Ct_op=1, #local Ct
                        Cp_op=1, #local Cp
                        u_lim=U_LIM,
                        cross_ts=True,ex=True) #INCLUDE cross terms
DUff_ijk = num_Fs.DUff_ijk #normalised DU / {U_\infty}

wavUff_j =  np.sum(P_i[:,None]*(1)*(np.sum(DUff_ijk,axis=2)),axis=0)
Uff2_ij_a = (1)*(np.sum(DUff_ijk,axis=2))**2
wavUff2_j_a = np.sum(P_i[:,None]*Uff2_ij_a,axis=0)

#then without the cross terms
powj_b,_,_= num_Fs(U_i,P_i,theta_i,
                   layout,
                   plot_points,
                   turb,
                   K=K,
                   RHO=1.225,
                   Ct_op=1, #local Ct
                   Cp_op=1, #local Cp
                   u_lim=U_LIM,
                   cross_ts=False,cube_term=False,ex=True) #EXCLUDE cross terms
#again the squared flow field (you can reuse DUff_ijk)
Uff2_ij_b = (1)*(np.sum(DUff_ijk**2,axis=2)) #U_i[:,None]**3
wavUff2_j_b = np.sum(P_i[:,None]*Uff2_ij_b,axis=0)
#wavUff2_j_b = np.sum(P_i[:,None]*np.sum(DUff_ijk**2,axis=2),axis=0)
#%% then plot all of that

from utilities.helpers import pce
from utilities.plotting_funcs import set_latex_font
set_latex_font() #set latex font ...

props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)

def nice_polar_plot(fig,gs,x,y,ann_txt,bar=True):
    ax = fig.add_subplot(gs,projection='polar')
    if bar:
        ax.bar(x,y,color='grey',linewidth=1,width=2*np.pi/72)
    else:
        ax.plot(x,y,color='black',linewidth=1)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.annotate(ann_txt, xy=(0.4,0.75), ha='center', va='bottom',color='black',xycoords='axes fraction',rotation='vertical',bbox=props)
    ax.spines['polar'].set_visible(False)

    return None

def arrow_plot(gs):
    #this is an unused function that shows an arrow for a single direction (not just "strongly single" - an actual impulse) wind rose
    ax = fig.add_subplot(gs[0,0])
    ax.set(xlim=(0,10),ylim=(0,10))
    ax.arrow(2,5,6,0, head_width=0.5, head_length=0.5, fc='k', ec='k')
    text = '$U_\infty =' +str(U_inf) + 'm/s^{-1}$'
    ax.text(5,3.4, text, size=8, ha="center", va="top",bbox=dict(boxstyle="square",ec='w', fc='w',pad=0,mutation_aspect=1.2))
    ax.axis('off')
    return None
  
from matplotlib import cm
def nice_composite_plot_v02(fig,gs1,gs2,Z1,X,Y,Z2,xt,yt,tpce):
    ax = fig.add_subplot(gs1)
    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-10)
    ax.tick_params(axis='both', which='both', pad=0,length=3)
    ax.tick_params(axis='x', which='both', pad=2)
    ax.set(xlim=(-XPAD, SPACING*(NT-1)+XPAD), ylim=(-YPAD, YPAD))
    #contourf
    cf_grey = ax.contourf(X,Y,Z1,cmap=cm.gray,levels=50)
    #scatter plot
    color_list = plt.cm.coolwarm(np.linspace(0, 1, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(color_list)
    sp = ax.scatter(xt,yt,c=Z2,cmap=cmap,marker='x',s=10)
    cax = fig.add_subplot(gs2)
    cb = fig.colorbar(sp, cax=cax, orientation='horizontal',format='%.3g')
    #number annotation
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)
    for i in range(NT): #label each turbine
        an_text = str(i+1)
        ax.annotate(an_text, xy=(xt[i],yt[i]-1), ha='center', va='top',color='black',bbox=props,fontsize=6,xycoords='data')
        
    #colorbar things
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(5))


    annotation_txt = f'approx. farm aep (\%error) : ${np.sum(powj_b):.2f}MW({tpce:+.2f}\\%)$'  
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  
    ax.annotate(annotation_txt, xy=(0.99,0.97), ha='right', va='top',color='black',bbox=props,xycoords='axes fraction',fontsize=9)

    cax.set_xlabel('Turbine error in AEP / \%',labelpad=3)

    ax.xaxis.set_ticks_position('none')
    ax.set_xticklabels([])
        
    return cf_grey,ax

def no_jumps_plot(x,y,ax,threshold,ls='-',color='black'):

    # Segments to be plotted
    segments = []
    current_segment = [[], []]
    # Loop through the data and create segments
    for i in range(1, len(y)):
        # Add the previous point to the current segment
        current_segment[0].append(x[i - 1])
        current_segment[1].append(y[i - 1])

        # If the jump is larger than the threshold, store the current segment and start a new one
        if abs(y[i] - y[i - 1]) > threshold:
            segments.append(current_segment)
            current_segment = [[], []]

    # Add the last segment
    current_segment[0].append(x[-1])
    current_segment[1].append(y[-1])
    segments.append(current_segment)

    # Plot the segments
    for segment in segments[1:]:
        ax.plot(segment[0], segment[1],color=color,lw=1,ls=ls)
    return None

def nice_plot2(fig,gs):
    #a "nice" cartesian plot
    ax = fig.add_subplot(gs)
    y0,y1,y2, = wavUff_j.reshape(xx.shape),wavUff2_j_a.reshape(xx.shape),wavUff2_j_b.reshape(xx.shape)
    label0 = "$\sum  \\frac{\\Delta U}{U_\\infty} $"
    label1 = "$\sum \\left( \\frac{\\Delta U}{U_\\infty} \\right)^2 $"
    label2 = "$\\left( \sum  \\frac{\\Delta U}{U_\\infty} \\right)^2 $"
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='grey', lw=1,ls='--',         label=label0),Line2D([0], [0], color='black', lw=1,ls='-',label=label2), Line2D([0], [0], color='black', lw=1,ls='-.',label=label1)
    ]
    
    threshold1 = np.max(y1[YRES//2,:])/10
    no_jumps_plot(xx[YRES//2,:],y0[YRES//2,:],ax,threshold1,ls='--',color='grey')
    no_jumps_plot(xx[YRES//2,:],y1[YRES//2,:],ax,threshold1,ls='-')
    no_jumps_plot(xx[YRES//2,:],y2[YRES//2,:],ax,threshold1,ls='-.')
    
    ax.set(xlim=(-XPAD, SPACING*(NT-1)+XPAD)) #ylim=(0,1.1*np.max(y1[YRES//2,:]))
    ax.legend(handles=legend_elements)

    return ax

def scale_cb(cax,c,d,labels=True):
    pos = cax.get_position()
    vmax = np.max(Uwff_a)
    vmin = np.min(Uwff_a)
    a = pos.x0 + (pos.width)*(vmin-d)/(c-d)
    b = pos.width*(vmax-vmin)/(c-d) 
    new_pos = [a, pos.y0, b, pos.height]
    cax.set_position(new_pos)
    from matplotlib.cm import ScalarMappable
    # I know this is bloated, I don't like it either
    sm = ScalarMappable(cmap=cm.gray, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal',format='%.3g')
    if not labels:
        cb.ax.xaxis.set_ticks_position('none')
        cb.ax.set_xticklabels([])
    return None

def connector_lines(fig,ax1,ax2):
    #Make the expansion lines
    from matplotlib.patches import ConnectionPatch
    for x in xt:
        con = ConnectionPatch(xyA=(x,-YPAD), xyB=(x,0), coordsA='data', coordsB='data', axesA=ax1, axesB=ax2,color='grey', ls='-',lw=1,alpha=0.5)
        fig.add_artist(con)
    return None

def ill_cb2(cf,gs): #illustrative colourbar on zoomed IN
    cax = fig.add_subplot(gs)
    fig.colorbar(cf,cax=cax, orientation='horizontal',format=f'%.3g')
    #scale_cb(cax,CP_UPPER,CP_LOWER,labels=False) 
    cax.set_xlabel('$\overline{U_w}$ / $ms^{-1}$',labelpad=0)
    return None

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
br = 20
gs = GridSpec(14, 3, height_ratios=[br,br,br,1,1,1,1,1,1,1,1,1,1,1],wspace=0.2,hspace=0.2)
fig = plt.figure(figsize=(5,5.7), dpi=200)

vmax = np.max(Uwff_a)
vmin = np.min(Uwff_a)

#first is the wind rose
nice_polar_plot(fig,gs[0,0],theta_i,U_i*P_i,"$P(\\theta)U_\infty(\\theta)$")
nice_polar_plot(fig,gs[0,1],theta_i,P_i,"$P(\\theta)$",bar=False)
nice_polar_plot(fig,gs[0,2],theta_i,U_i,"$U_\infty(\\theta)$",bar=False)

#next is the flow field contourf
z2 = pce(powj_a,powj_b)
tpce = pce(np.sum(powj_a),np.sum(powj_b))
cf,ax1 = nice_composite_plot_v02(fig,gs[1,:],gs[5,:],Uwff_a.reshape(xx.shape),xx,yy,z2,xt,yt,tpce)

#next is the zoomed in thrust coefficient curve
ax2 = nice_plot2(fig,gs[2,:])
ill_cb2(cf,gs[10,:])

connector_lines(fig,ax1,ax2) #the vertical grey lines

if SAVE_FIG:
    from pathlib import Path

    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "fig images"
    fig_name = f"Fig_XtermSingle_{U_inf}ms_{np.rad2deg(theta_i[non_zero_bin])*10:.0f}.png"
    image_path = fig_dir / fig_name
    path_plus_name = fig_dir / fig_name

    plt.savefig(path_plus_name, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")


