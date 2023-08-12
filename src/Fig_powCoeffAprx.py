#%% 
# Produce the figure to demonstrate the limitations of the POWER coefficient approximation: C_p(U_w) \approx C_p(U_\infty)

%load_ext autoreload
%autoreload 2

import numpy as np

west = False #westerly direction if true

NT = 5 #number of turbines 
SPACING = 7 #spacing in diameters
BINS = 360
U_LIM = 4
XPAD = 7 #padding around the turbines for the contour plot
YPAD = 7

CP_LOWER = 0
CP_UPPER = 27

SAVE_FIG = True

def get_layout():
    #linear layout
    xt = np.arange(0,NT*SPACING,SPACING)
    yt = np.zeros_like(xt)
    return xt,yt,np.column_stack((xt,yt))

from scipy.stats import vonmises

#the Easterly portion of the wind rose
def east_wr(U_inf,kappa):
    U_i = np.zeros(BINS)
    U_i[0:180] = U_inf #0 to pi is U_inf
    mu1 = (1/2)*np.pi # westerly direction
    P_i = vonmises.pdf(theta_i, kappa, loc=mu1) 
    P_i = P_i/np.sum(P_i) #normalise for discrete distribution
    return U_i,P_i 

#the Westerly portion of the wind rose
def west_wr(U_inf,kappa):
    U_i = np.zeros(BINS)
    U_i[180:] = U_inf #pi to 2pi is U_inf
    mu2 = (3/2)*np.pi #easterly direction
    P_i = vonmises.pdf(theta_i, kappa, loc=mu2)
    P_i = P_i/np.sum(P_i) #normalise for discrete distribution
    return U_i,P_i 

from utilities.turbines import iea_10MW
turb = iea_10MW()

theta_i = np.linspace(0,2*np.pi,BINS,endpoint=False)
kappa = 8.0

if west:
    a_0 = 6
    U_i,P_i = west_wr(a_0,kappa)
else: #otherwise easterly
    a_0 = 14
    U_i,P_i = east_wr(a_0,kappa)

from utilities.helpers import linear_layout,rectangular_domain,pce
xt,yt,layout = linear_layout(NT,SPACING)
xx,yy,plot_points,xlims,ylims = rectangular_domain(layout,xr=300)

from utilities.AEP3_functions import num_F_v02
#reference AEP first (Cp_op == 1 means Cp(U_w))
aep_a,Uwt_ja,Uwff_ja= num_F_v02(U_i,P_i,theta_i,
                       layout,
                       plot_points,
                       turb,
                       RHO=1.225,K=0.025,
                       u_lim=U_LIM,Ct_op=1,cross_ts=True,ex=True,Cp_op=1)
#then Cp_op == 2 means Cp(U_\infty)
aep_b,Uwt_jb,_      = num_F_v02(U_i,P_i,theta_i,
                       layout,
                       plot_points, 
                       turb,
                       RHO=1.225,K=0.025,
                       u_lim=U_LIM,Ct_op=1,cross_ts=True,ex=True,Cp_op=2)

#the rest is a load leg work plotting the results

#set font
from utilities.plotting_funcs import set_latex_font
set_latex_font() #set latex font 

def nice_polar_plot(fig,gs,x,y,ann_txt,bar=True,ylim=None):
    ax = fig.add_subplot(gs,projection='polar')
    if bar:
        ax.bar(x,y,color='grey',linewidth=1,width=2*np.pi/72)
    else:
        ax.plot(x,y,color='black',linewidth=1)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=3)
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)
    ax.annotate(ann_txt, xy=(0.4,0.75), ha='center', va='bottom',color='black',xycoords='axes fraction',rotation='vertical',bbox=props)
    ax.spines['polar'].set_visible(False)
    if ylim is not None:
        ax.set(ylim=(None,ylim))
    return None
      
from matplotlib import cm
def nice_composite_plot_v02(fig,gs1,gs2,Z1,X,Y,Z2,xt,yt,tpce):
    ax = fig.add_subplot(gs1)
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
    ax.set(xlim=(-XPAD, SPACING*(NT-1)+XPAD), ylim=(-YPAD, YPAD))

    annotation_txt = f'farm aep (\%error) : ${np.sum(aep_b):.2f}MW({tpce:+.2f}\\%)$'  
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  
    ax.annotate(annotation_txt, xy=(0.99,0.97), ha='right', va='top',color='black',bbox=props,xycoords='axes fraction',fontsize=9)

    cax.set_xlabel('Turbine error in AEP / \%',labelpad=-1)
        
    return cf_grey

def horizontal_line(ax,x,y,text,ha,va,xoff,yoff):
    ax.axhline(y=y, color='grey', ls=':',lw=1)
    annotation_txt = '$C_p(U_\\infty)'+f'={y:.2f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)
    ax.annotate(annotation_txt, xy=(x,y), ha=ha, va=va,color='black',bbox=props,xycoords='data',fontsize=8,xytext=(xoff, yoff), textcoords='offset points')  
    return None

def nice_plot1(fig,gs,Uw_js):
    #a "nice" plot of the turbine power coefficient curve
    ax = fig.add_subplot(gs)
    xs = np.linspace(CP_LOWER,CP_UPPER,200)
    ax.plot(xs,turb.Cp_f(xs),color='grey',linewidth=1)
    ax.scatter(Uw_js,turb.Cp_f(Uw_js),marker='x',s=10,color='black')
    ax.set(xlim=(CP_LOWER,CP_UPPER),ylabel='$C_p$',ylim=(0,None),xlabel='Wake Velocity $U_w$ / $ms^{-1}$')#
    horizontal_line(ax,0,turb.Cp_f(a_0),None,'left','top',2,text1_yoff)
    return ax

def nice_plot2(fig,gs,Uw_js):
    #a "nice" plot of the zoomed IN power coefficient bit
    ax = fig.add_subplot(gs)
    xmin,xmax = np.min(Uwff_ja),np.max(Uwff_ja)
    xrng = 0.1*(xmax - xmin)
    cp_arr = np.append(turb.Cp_f(Uwff_ja),turb.Cp_f(a_0))
    ymax,ymin = np.max(cp_arr),np.min(cp_arr)
    yrng = 0.2*(ymax-ymin)
    xs = np.linspace(xmin-xrng,xmax+xrng,200)
    ax.set(xlim=(xmin-xrng,xmax+xrng),ylim=(ymin-yrng,ymax+yrng),ylabel='$C_p$',xticks=[], xticklabels=[])
    ax.plot(xs,turb.Cp_f(xs),color='grey',linewidth=1)
    ax.scatter(Uw_js,turb.Cp_f(Uw_js),marker='x',s=10,color='black')
    'an'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)
    offset = (np.max(turb.Cp_f(Uw_js)) - np.min(turb.Cp_f(Uw_js))) * 0.1
    for i in range(NT): #label each turbine
            Uw = Uwt_ja[i]
            ax.annotate(str(i+1), xy=(Uw,turb.Cp_f(Uw)-offset), ha='center', va='top',color='black',bbox=props,fontsize=6,xycoords='data')
    horizontal_line(ax,xmin-xrng,turb.Cp_f(a_0),None,'left',text2_align,2,text2_yoff)
    return ax,ymax+yrng

def scale_cb(cax,c,d,labels=True):
    pos = cax.get_position()
    vmax = np.max(Uwff_ja)
    vmin = np.min(Uwff_ja)
    a = pos.x0 + (pos.width)*(vmin-d)/(c-d)
    b = pos.width*(vmax-vmin)/(c-d) 
    new_pos = [a, pos.y0, b, pos.height]
    cax.set_position(new_pos)
    from matplotlib.cm import ScalarMappable
    # I know this is bloated, I don't like it either
    sm = ScalarMappable(cmap=cm.gray, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    import matplotlib.ticker as ticker
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal',format='%.3g',ticks=ticker.LinearLocator(5))
    if not labels:
        cb.ax.xaxis.set_ticks_position('none')
        cb.ax.set_xticklabels([])
    return None

def expansion_lines(fig,ax1,ax2,y2):
    #Make the expansion lines
    from matplotlib.patches import ConnectionPatch
    xrng = 0.1*(vmax-vmin)
    con = ConnectionPatch(xyA=(vmin-xrng,0), xyB=(0,1), coordsA='data', coordsB='axes fraction', axesA=ax1, axesB=ax2,color='black', ls='-',lw=0.5)
    fig.add_artist(con)
    con = ConnectionPatch(xyA=(vmax+xrng,0), xyB=(1,1), coordsA='data', coordsB='axes fraction', axesA=ax1, axesB=ax2,color='black', ls='-',lw=0.5)
    fig.add_artist(con)
    return None

def ill_cb1(cf,gs): #illustrative colourbar on zoomed IN
    cax = fig.add_subplot(gs)
    fig.colorbar(cf,cax=cax, orientation='horizontal',format='%.3g')
    scale_cb(cax,CP_UPPER,CP_LOWER,labels=False) 
    return None

def ill_cb2(cf,gs): #illustrative colourbar on zoomed IN
    cax = fig.add_subplot(gs)
    fig.colorbar(cf,cax=cax, orientation='horizontal',format='%.3g')
    rng = 0.1*(np.max(Uwff_ja) - np.min(Uwff_ja))
    scale_cb(cax,np.max(Uwff_ja)+rng,np.min(Uwff_ja)-rng) 
    cax.set_xlabel('Wake Velocity $U_w$ / $ms^{-1}$',labelpad=0)
    return None

#a bunch of plotting faff 
if west:
    aU_i = np.append(U_i,0)
    atheta_i = np.append(theta_i,0)
    #the Cp(U_\infty) label needs *slightly* offsetting differently for each case
    text1_yoff = -6 #this is so the label doesn't overlap the points
    text2_align='top' #ditto
    text2_yoff = -2 #ditto
else:
    aU_i = np.append(0,U_i) #this is so the polar plot shows a complete semi-circle
    atheta_i = np.append(0,theta_i) 
    text1_yoff = -2
    text2_align='bottom'
    text2_yoff = 2
    
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
br = 20
gs = GridSpec(8, 3, height_ratios=[br,br,1,1,br,1,br,1],wspace=0.2,hspace=0.4)
fig = plt.figure(figsize=(5,7), dpi=200)

vmax = np.max(Uwff_ja)
vmin = np.min(Uwff_ja)

#first is the wind roses
nice_polar_plot(fig,gs[0,0],theta_i,U_i*P_i,"$P(\\theta)U_\infty(\\theta)$")
nice_polar_plot(fig,gs[0,1],theta_i,P_i,"$P(\\theta)$",bar=False)
#using appended values for the last plot
nice_polar_plot(fig,gs[0,2],atheta_i,aU_i,"$U_\infty(\\theta)$",bar=False,ylim=15)

#next is the flow field contourf
z2 = pce(aep_a,aep_b)
tpce = pce(np.sum(aep_a),np.sum(aep_b))
cf = nice_composite_plot_v02(fig,gs[1,:],gs[2,:],Uwff_ja.reshape(xx.shape),xx,yy,z2,xt,yt,tpce)

#next is the power coefficient curve
ax1 = nice_plot1(fig,gs[4,:],Uwt_ja)

#next is the ZOOMED thrust coefficient curve
ax2,y2 = nice_plot2(fig,gs[6,:],Uwt_ja)
expansion_lines(fig,ax1,ax2,y2)

ill_cb1(cf,gs[5,:])
ill_cb2(cf,gs[7,:])

if SAVE_FIG:
    from pathlib import Path

    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "fig images"
    fig_name = f"Fig_powCoeffAprx_{a_0}ms{'W'*west}{'E'*(1-west)}.png"
    image_path = fig_dir / fig_name

    plt.savefig(image_path, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")
