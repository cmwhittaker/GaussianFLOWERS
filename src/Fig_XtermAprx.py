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
U_inf1 = 10 #12 or 5
U_inf2 = 12
U_LIM = 4

CP_LOWER = 0
CP_UPPER = 27

SAVE_FIG = False

EXTENT = 14

RES = 201

def get_layout():
    xt = np.arange(0,NT*SPACING,SPACING)
    yt = np.zeros_like(xt)
    return xt,yt,np.column_stack((xt,yt))

def rectangular_domain(r=RES):
    xx,yy = np.meshgrid(np.linspace(-XPAD,(NT-1)*SPACING+XPAD,r),np.linspace(-YPAD,YPAD,r))
    return xx,yy,np.column_stack((xx.reshape(-1),yy.reshape(-1)))

def get_real_wr():
    from distributions_vC05 import wind_rose
    wr = wind_rose(bin_no_bins=BINS,custom=None,site=6,Cp_f=turb.Cp_f)
    U_i = wr.avMagnitude
    P_i = wr.frequency
    return U_i,P_i

def custom_wr():
    U_i = U_inf1*np.ones(BINS)
    from scipy.stats import vonmises
    kappa = 8.0  # concentration parameter
    mu1 = (3/2)*np.pi # mean direction (in radians)
    P_i = vonmises.pdf(theta_i, kappa, loc=mu1)
    P_i = P_i/np.sum(P_i) #normalise for discrete distribution
    return U_i,P_i 

from turbines_v01 import iea_10MW
turb = iea_10MW()
Ct_f = turb.Ct_f
Cp_f = turb.Cp_f

U_i = np.ones(360,)*U_inf1
P_i = np.zeros_like(U_i)
P_i[270] = 1 #blows from a single wind direction
theta_i = np.linspace(0,2*np.pi,BINS,endpoint=False)
#U_i,P_i = get_real_wr()
U_i,P_i = custom_wr()

xt,yt,layout = get_layout()
xx,yy,plot_points = rectangular_domain()

from AEP3_3_functions import gen_local_grid_v01C
#slightly modified numerical function
def num_F_v01(U_i,P_i,theta_i,
          r_jk,theta_jk,
          turb,
          RHO=1.225,K=0.025,
          u_lim=None,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True):
    #function to show the different effects of the many assumptions
    #i:directions,j:turbines,k:turbines in superposistion
    #invalid: specific an invalid radius
    #cross_t: cross terms in cubic expansion
    #sml_a: small_angle approximation
    #local_cp:local power coeff (or global)
    #(var_ct: ct is fixed externally with a lambda function if wanted)
    def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
        
        if u_lim is not None:
            lim = u_lim
        else:
            lim = (np.sqrt(Ct/8)-ep)/K
            lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be <0 but there is numerical artifacting in rsin(theta) )
        
        theta = theta + np.pi #the wake lies opposite!
        if ex: #use full 
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*np.cos(theta)+ep)**2))))*(np.exp(-(r*np.sin(theta))**2/(2*(K*r*np.cos(theta)+ep)**2)))
            deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit  
        else: #otherwise use small angle approximations
            theta = np.mod(theta-np.pi,2*np.pi)-np.pi
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*1+ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+ep)**2)))          
            deltaU_by_Uinf = np.where(r>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
            return deltaU_by_Uinf      
        
        return deltaU_by_Uinf  

    Ct_f = turb.Ct_f
    Cp_f = turb.Cp_f
    A = turb.A

    if not var_Ct: #use the Fixed (Weight averaged) Ct?     
        WAV_CT = np.sum(Ct_f(U_i)*P_i)
        Ct_f = lambda x: WAV_CT

    #when plot_points == layout it finds wake at the turbine posistions
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None]

    r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)
    ct_ijk = Ct_f(U_i)[...,None,None]*np.ones((r_jk.shape[0],r_jk.shape[1]))[None,...] #this is a dirty way of repeating along 2 axis

    def soat(a): #Sum over Axis Two
        return np.sum(a,axis=2)

    DU_by_Uinf_ijk = deltaU_by_Uinf_f(r_ijk,theta_ijk,ct_ijk,K) #deltaU_by_Uinf as a function
    if cross_ts: #INcluding cross terms
        Uw_ij_cube = (U_i[:,None]*(1-np.sum(DU_by_Uinf_ijk,axis=2)))**3
    else: #EXcluding cross terms (soat = Sum over Axis Two (third axis!)
        Uw_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(DU_by_Uinf_ijk) + 3*soat(DU_by_Uinf_ijk**2) - soat(DU_by_Uinf_ijk**3))

    Uw_ij = (U_i[:,None]*(1-np.sum(DU_by_Uinf_ijk,axis=2)))
    if lcl_Cp: #power coeff based on local wake velocity
        Cp_ij = Cp_f(Uw_ij)
    else: #power coeff based on global inflow U_infty
        Cp_ij = Cp_f(U_i)[:,None]

    #sum over wind directions (i) (this is the weight-averaging)
    if avCube: #directly find the average of the cube velocity
        pow_j = 0.5*A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uw_ij_cube),axis=0)/(1*10**6)
    else: #the old way of cubing the weight-averaged field
        WAV_CP = np.sum(Cp_f(U_i)*P_i) #frequency-weighted av Cp on global
        pow_j = 0.5*A*RHO*WAV_CP*np.sum(P_i[:,None]*Uw_ij**3,axis=0)/(1*10**6)

    Uw_j = np.sum(P_i[:,None]*Uw_ij,axis=0) #flow field
    a1 = np.sum(P_i[:,None]*np.sum(DU_by_Uinf_ijk,axis=2)**2,axis=0)
    a2 = np.sum(P_i[:,None]*np.sum(DU_by_Uinf_ijk**2,axis=2),axis=0)
    return pow_j,Uw_j,a1,a2 #power(mw)/wake velocity 

r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
#aep first
aep_a,Uw_ja,_,_= num_F_v01(U_i,P_i,theta_i,
                 r_jk,theta_jk,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)
aep_b,_,_,_  = num_F_v01(U_i,P_i,theta_i,
                 r_jk,theta_jk,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=False,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)
#then the flow field
r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
_,ff_a,a1,a2  = num_F_v01(U_i,P_i,theta_i,
                 r_jk,theta_jk,
                 turb,
                 RHO=1.225,K=0.025,
                 u_lim=U_LIM,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)

#%%
def pce(exact,approx):
    return 100*(exact-approx)/exact

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
    #first row, first column is the arrow
    ax = fig.add_subplot(gs[0,0])
    ax.set(xlim=(0,10),ylim=(0,10))
    ax.arrow(2,5,6,0, head_width=0.5, head_length=0.5, fc='k', ec='k')
    text = '$U_\infty =' +str(U_inf1) + 'm/s^{-1}$'
    ax.text(5,3.4, text, size=8, ha="center", va="top",bbox=dict(boxstyle="square",ec='w', fc='w',pad=0,mutation_aspect=1.2))
    ax.axis('off')
    return None
  
from matplotlib import cm
def nice_composite_plot_v02(fig,gs1,gs2,Z1,X,Y,Z2,xt,yt,tpce):
    ax = fig.add_subplot(gs1)
    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-15)
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

    annotation_txt = f'error in farm aep: ${tpce:+.2f}$\\%'  
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  
    ax.annotate(annotation_txt, xy=(0.99,0.97), ha='right', va='top',color='black',bbox=props,xycoords='axes fraction',fontsize=9)

    cax.set_xlabel('Turbine error in AEP / \%',labelpad=3)

    ax.xaxis.set_ticks_position('none')
    ax.set_xticklabels([])
        
    return cf_grey,ax

def no_jumps_plot(x,y,ax,threshold,ls='-'):

    # Threshold for the jump in y

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
        ax.plot(segment[0], segment[1],color='black',lw=1,ls=ls)
    return None

def nice_plot2(fig,gs):
    #a "nice" cartesian plot
    ax = fig.add_subplot(gs)
    y1,y2 = a1.reshape(xx.shape),a2.reshape(xx.shape)
    label1 = "$\sum \\left( \\frac{\\Delta U}{U_\\infty} \\right)^2 $"
    label2 = "$\\left( \sum  \\frac{\\Delta U}{U_\\infty} \\right)^2 $"
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='black', lw=1,ls='-',label=label2),
                       Line2D([0], [0], color='black', lw=1,ls='-.',label=label1)]
    threshold = np.max(y1[RES//2,:])/10
    no_jumps_plot(xx[RES//2,:],y1[RES//2,:],ax,threshold,ls='-')
    no_jumps_plot(xx[RES//2,:],y2[RES//2,:],ax,threshold,ls='-.')
    ax.set(xlim=(-XPAD, SPACING*(NT-1)+XPAD),ylim=(0,1.1*np.max(y1[RES//2,:])))
    ax.legend(handles=legend_elements)

    return ax

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

def connector_lines(fig,ax1,ax2):
    #Make the expansion lines
    from matplotlib.patches import ConnectionPatch
    for x in xt:
        con = ConnectionPatch(xyA=(x,-YPAD), xyB=(x,0), coordsA='data', coordsB='data', axesA=ax1, axesB=ax2,color='grey', ls='-',lw=1,alpha=0.5)
        fig.add_artist(con)
    return None

def ill_cb2(cf,gs): #illustrative colourbar on zoomed IN
    cax = fig.add_subplot(gs)
    fig.colorbar(cf,cax=cax, orientation='horizontal',format='%.3g')
    #scale_cb(cax,CP_UPPER,CP_LOWER,labels=False) 
    cax.set_xlabel('Wake Velocity $U_w$ / $ms^{-1}$',labelpad=0)
    return None

#plt.fill_between(x, y, where=(x >= x_start) & (x <= x_end), color='gray', alpha=0.5)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
br = 20
gs = GridSpec(14, 3, height_ratios=[br,br,br,1,1,1,1,1,1,1,1,1,1,1],wspace=0.2,hspace=0.1)
fig = plt.figure(figsize=(5,7), dpi=200)

vmax = np.max(ff_a)
vmin = np.min(ff_a)

#first is the wind rose
nice_polar_plot(fig,gs[0,0],theta_i,U_i*P_i,"$P(\\theta)U_\infty(\\theta)$")
nice_polar_plot(fig,gs[0,1],theta_i,P_i,"$P(\\theta)$",bar=False)
nice_polar_plot(fig,gs[0,2],theta_i,U_i,"$U_\infty(\\theta)$",bar=False)

#next is the flow field contourf
z2 = pce(aep_a,aep_b)
tpce = pce(np.sum(aep_a),np.sum(aep_b))
cf,ax1 = nice_composite_plot_v02(fig,gs[1,:],gs[5,:],ff_a.reshape(xx.shape),xx,yy,z2,xt,yt,tpce)

#next is the zoomed in thrust coefficient curve
ax2 = nice_plot2(fig,gs[2,:])
ill_cb2(cf,gs[10,:])

connector_lines(fig,ax1,ax2)

if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+"_"+str(U_inf1)+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')

    print("figure saved")
