#%% set font!
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

#%%generate the data
import numpy as np

NT = 5 #number of turbines 
SPACING = 7 
XPAD = 7 #X pad
YPAD = 7
BINS = 360
a_0 = 5 #12 or 5
U_LIM = 4

CP_LOWER = 0
CP_UPPER = 27

SAVE_FIG = True

def get_layout():
    xt = np.arange(0,NT*SPACING,SPACING)
    yt = np.ones_like(xt)
    return xt,yt,np.column_stack((xt,yt))

def rectangular_domain(r=100):
    xx,yy = np.meshgrid(np.linspace(-XPAD,(NT-1)*SPACING+XPAD,r),np.linspace(-YPAD,YPAD,r))
    return xx,yy,np.column_stack((xx.reshape(-1),yy.reshape(-1)))

def get_real_wr():
    from distributions_vC05 import wind_rose
    wr = wind_rose(bin_no_bins=BINS,custom=None,site=6,Cp_f=turb.Cp_f)
    U_i = wr.avMagnitude*0.4
    P_i = wr.frequency
    return U_i,P_i

def get_von_mises_wr():
    U_i = np.ones(BINS)*a_0
    from scipy.stats import vonmises
    kappa = 8.0  # concentration parameter
    mu = (3/2)*np.pi # mean direction (in radians)
    P_i = vonmises.pdf(theta_i, kappa, loc=mu)
    P_i = P_i/np.sum(P_i) #normalise for discrete distribution
    return U_i,P_i 

from turbines_v01 import iea_10MW
turb = iea_10MW()
Ct_f = turb.Ct_f
Cp_f = turb.Cp_f

U_i = np.ones(360,)*a_0
P_i = np.zeros_like(U_i)
P_i[270] = 1 #blows from a single wind direction
theta_i = np.linspace(0,2*np.pi,BINS,endpoint=False)
#U_i,P_i = get_real_wr()
U_i,P_i = get_von_mises_wr()

xt,yt,layout = get_layout()
xx,yy,plot_points = rectangular_domain()

from AEP3_3_functions import num_F,gen_local_grid_v01C
r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
#aep first
aep_a,Uw_ja= num_F(U_i,P_i,theta_i,
                 r_jk,theta_jk,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)
aep_b,_  = num_F(U_i,P_i,theta_i,
                 r_jk,theta_jk,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=True,ex=True,lcl_Cp=False,avCube=True,var_Ct=True)
#then the flow field
r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
_,ff_a   = num_F(U_i,P_i,theta_i,
                 r_jk,theta_jk,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)

#% plot the data

def pce(exact,approx):
    return 100*(exact-approx)/exact

def nice_polar_plot(fig,gs,x,y):
    ax = fig.add_subplot(gs[0,0],projection='polar')
    ax.bar(x,y,color='grey',linewidth=1,width=2*np.pi/72)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.annotate("$P(\\theta)$", xy=(0.4,0.75), ha='center', va='bottom',color='black',xycoords='axes fraction',rotation='vertical')
    text = "$U_\infty=" + str(a_0) + 'm/s^{-1}$'
    ax.text(0, 0, text, ha='left',transform=ax.transAxes,color='black')
    ax.spines['polar'].set_visible(False)

    return None

def arrow_plot(gs):
    #first row, first column is the arrow
    ax = fig.add_subplot(gs[0,0])
    ax.set(xlim=(0,10),ylim=(0,10))
    ax.arrow(2,5,6,0, head_width=0.5, head_length=0.5, fc='k', ec='k')
    text = '$U_\infty =' +str(a_0) + 'm/s^{-1}$'
    ax.text(5,3.4, text, size=8, ha="center", va="top",bbox=dict(boxstyle="square",ec='w', fc='w',pad=0,mutation_aspect=1.2))
    ax.axis('off')
    return None
  
from matplotlib import cm
def nice_composite_plot_v02(fig,gs,Z1,X,Y,Z2,xt,yt,tpce):
    ax = fig.add_subplot(gs[0,1:])
    xticks = ax.xaxis.get_major_ticks()
    xticks[5].set_visible(False)
    ax.set_xlabel('$x/d_0$',labelpad=-9)
    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-15)
    ax.tick_params(axis='both', which='both', pad=0,length=3)
    ax.tick_params(axis='x', which='both', pad=2)
    #contourf
    cf_grey = ax.contourf(X,Y,Z1,cmap=cm.gray,levels=50)
    #scatter plot
    color_list = plt.cm.coolwarm(np.linspace(0, 1, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(color_list)
    sp = ax.scatter(xt,yt,c=Z2,cmap=cmap,marker='x',s=10)
    cax = fig.add_subplot(gs[1,1:])
    cb = fig.colorbar(sp, cax=cax, orientation='horizontal',format='%.3g')
    #number annotation
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)
    for i in range(NT): #label each turbine
        an_text = str(i+1)
        ax.annotate(an_text, xy=(xt[i],yt[i]-1), ha='center', va='top',color='black',bbox=props,fontsize=6,xycoords='data')
        
    #colorbar things
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set(xlim=(-XPAD, SPACING*(NT-1)+XPAD), ylim=(-YPAD, YPAD))

    annotation_txt = f'error in farm aep: ${tpce:+.2f}$\\%'  
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  
    ax.annotate(annotation_txt, xy=(0.99,0.97), ha='right', va='top',color='black',bbox=props,xycoords='axes fraction',fontsize=9)

    cax.set_xlabel('Turbine error in AEP / \%',labelpad=-1)
        
    return cf_grey

def nice_plot1(fig,gs,Uw_js):
    #a "nice" plot of the turbine power coefficient curve
    ax = fig.add_subplot(gs[3,1:])
    xs = np.linspace(CP_LOWER,CP_UPPER,200)
    ax.plot(xs,Cp_f(xs),color='grey',linewidth=1)
    ax.scatter(Uw_js,Cp_f(Uw_js),marker='x',s=10,color='black')
    ax.set(xlim=(CP_LOWER,CP_UPPER),ylabel='$C_p$',ylim=(0,None))#
    ax.axvline(x=vmin, color='grey', ls=':',lw=1)
    ax.axvline(x=vmax, color='grey', ls=':',lw=1)
    return ax

def nice_plot2(fig,gs,Uw_js):
    #a "nice" plot of the zoomed IN power coefficient bit
    ax = fig.add_subplot(gs[5,1:])
    xmin,xmax = np.min(ff_a),np.max(ff_a)
    xrng = 0.1*(xmax - xmin)
    ymax,ymin = np.max(Cp_f(ff_a)),np.min(Cp_f(ff_a))
    yrng = 0.2*(ymax-ymin)
    xs = np.linspace(xmin-xrng,xmax+xrng,200)
    ax.set(xlim=(xmin-xrng,xmax+xrng),ylim=(ymin-yrng,ymax+yrng))
    ax.plot(xs,Cp_f(xs),color='grey',linewidth=1)
    ax.scatter(Uw_js,Cp_f(Uw_js),marker='x',s=10,color='black')
    'an'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)
    offset = (np.max(Cp_f(Uw_js)) - np.min(Cp_f(Uw_js))) * 0.1
    for i in range(NT): #label each turbine
            Uw = Uw_ja[i]
            ax.annotate(str(i+1), xy=(Uw,Cp_f(Uw)-offset), ha='center', va='top',color='black',bbox=props,fontsize=6,xycoords='data')
    ax.set(ylabel='$C_p$')
    ax.set_xticklabels([])
    return ax,ymax+yrng

def scale_cb(cax,c,d,labels=True):
    pos = cax.get_position()
    vmax = np.max(ff_a)
    vmin = np.min(ff_a)
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

def expansion_lines(fig,ax1,ax2,y2):
    #Make the expansion lines
    from matplotlib.patches import ConnectionPatch
    xrng = 0.1*(vmax-vmin)
    con = ConnectionPatch(xyA=(vmin-xrng,0), xyB=(vmin-xrng,y2), coordsA='data', coordsB='data', axesA=ax1, axesB=ax2,color='grey', ls=':',lw=1)
    fig.add_artist(con)
    con = ConnectionPatch(xyA=(vmax+xrng,0), xyB=(vmax+xrng,y2), coordsA='data', coordsB='data', axesA=ax1, axesB=ax2,color='grey', ls=':',lw=1)
    fig.add_artist(con)
    return None

def ill_cb1(cf): #illustrative colourbar on zoomed IN
    cax = fig.add_subplot(gs[4,1:])
    fig.colorbar(cf,cax=cax, orientation='horizontal',format='%.3g')
    scale_cb(cax,CP_UPPER,CP_LOWER,labels=False) 
    return None

def ill_cb2(cf): #illustrative colourbar on zoomed IN
    cax = fig.add_subplot(gs[6,1:])
    fig.colorbar(cf,cax=cax, orientation='horizontal',format='%.3g')
    rng = 0.1*(np.max(ff_a) - np.min(ff_a))
    scale_cb(cax,np.max(ff_a)+rng,np.min(ff_a)-rng) 
    cax.set_xlabel('Wake Velocity $U_w$ / $ms^{-1}$',labelpad=0)
    return None


#plt.fill_between(x, y, where=(x >= x_start) & (x <= x_end), color='gray', alpha=0.5)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
br = 20
gs = GridSpec(7, 4, height_ratios=[1.5*br,1,1,br,1,br,1],wspace=0.2,hspace=0.45)
fig = plt.figure(figsize=(6.7,5.5), dpi=200)

vmax = np.max(ff_a)
vmin = np.min(ff_a)

#first is the wind rose
nice_polar_plot(fig,gs,theta_i,U_i*P_i)

#next is the flow field contourf
z2 = pce(aep_a,aep_b)
tpce = pce(np.sum(aep_a),np.sum(aep_b))
cf = nice_composite_plot_v02(fig,gs,ff_a.reshape(xx.shape),xx,yy,z2,xt,yt,tpce)

#next is the power coefficient curve
ax1 = nice_plot1(fig,gs,Uw_ja)

#next is the zoomed in thrust coefficient curve
ax2,y2 = nice_plot2(fig,gs,Uw_ja)

expansion_lines(fig,ax1,ax2,y2)

ill_cb1(cf)
ill_cb2(cf)

if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+"_"+str(a_0)+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')

    print("figure saved")
