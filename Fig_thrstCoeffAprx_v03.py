#% This was a "working" version of 2 where I was correcting the coordinate systems. That is now all done and v4A is the functional version (which is now far more flexible)
#%% get data to plot
%load_ext autoreload
%autoreload 2
import numpy as np
SAVE_FIG = False
SPACING = 7
INVALID_R = 5
RESOLUTION = 100
EXTENT = 30

def rectangular_domain(extent,r=200):
    X,Y = np.meshgrid(np.linspace(-extent,extent,r),np.linspace(-extent,extent,r))
    return X,Y,np.column_stack((X.reshape(-1),Y.reshape(-1)))

def rectangular_layout(no_xt,s):
    n = no_xt//2
    xt = np.arange(-n,n+1,1)*s
    yt = np.arange(-n,n+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)
    return np.column_stack((Xt.reshape(-1),Yt.reshape(-1)))#just a single layout for now

def get_custom_layouts(s):
    layout_array = [[0 for j in range(cols)] for i in range(rows)]
    layout_array[:][0] = rectangular_layout(3,s)
    layout_array[:][1] = rectangular_layout(5,s)
    layout_array[:][2] = rectangular_layout(7,s)

    return layout_array

def get_wind_roses(no_bins):
    #hard-coded wind roses (this was to allow custom ones...)
    wr_list = []
    for i in range(rows): #for each row
        if i == 0: #the uniform wind rose
            wr = wind_rose(bin_no_bins=no_bins,custom=2,site=2,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        if i == 1: #site 6 has one strong direction 
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=3,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        if i == 2: #impulse wind rose (most extreme)
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=6,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        wr_list.append(wr)
    return wr_list

def F(invalid=None,cross_ts=True,sml_a=False,local_Cp=True):
    #function to show the difference caused by neglecting the cross terms. note: this is NOT the same as cubeAv_v4. The power coefficient is approximated using (directional dependant) global inflow.
    #i:directions,j:turbines,k:turbines in superposistion
    #invalid: specific an invalid radius
    #cross_t: cross terms in cubic expansion
    #sml_a: small_angle approximation
    #local_cp:local power coeff (or global)
    #(var_ct: ct is fixed externally with a lambda function if wanted)
    def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
        lim = (np.sqrt(Ct/8)-ep)/K
        lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be >0 but there is numerical artifacting in rsin(theta) )
        theta = theta + np.pi #the wake lies opposite!
        if INVALID_R is not None:
            lim = invalid
        if sml_a: #use small angle approximations
            theta = np.mod(theta-np.pi,2*np.pi)-np.pi
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*1+ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+ep)**2)))          
            deltaU_by_Uinf = np.where(r>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
            return deltaU_by_Uinf
        else: #use full 
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*np.cos(theta)+ep)**2))))*(np.exp(-(r*np.sin(theta))**2/(2*(K*r*np.cos(theta)+ep)**2)))
            deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit        
        
        return deltaU_by_Uinf

    def soat(a): #Sum over Axis Two
        return np.sum(a,axis=2)
    
    #when plot_points = layout it finds wake at the turbine posistions
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None]

    r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)
    ct_ijk = Ct_f(U_i)[...,None,None]*np.ones((r_jk.shape[0],r_jk.shape[1]))[None,...] #this is a dirty way of repeating along 2 axis

    DU_by_Uinf = deltaU_by_Uinf_f(r_ijk,theta_ijk,ct_ijk,K) #deltaU_by_Uinf DU_by_Uinf
    if cross_ts: #INcluding cross terms
        Uw_ij_cube = (U_i[:,None]*(1-np.sum(DU_by_Uinf,axis=2)))**3
    else: #EXcluding cross terms (soat = Sum over Axis Two (third axis!)
        Uw_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(DU_by_Uinf) + 3*soat(DU_by_Uinf**2) - soat(DU_by_Uinf**3))

    Uw_i = (U_i[:,None]*(1-np.sum(DU_by_Uinf,axis=2)))
    if local_Cp:
        Cp_i = Cp_f(Uw_i)
    else:
        Cp_i = Cp_f(U_i)

    pow_j = np.sum(P_i[:,None]*(0.5*A*RHO*Cp_i*Uw_ij_cube),axis=0)/(1*10**6)
    return pow_j #power of each turbine in mw

def empty2dPyarray(rows,cols): #create empty 2d python array
    return [[0 for j in range(cols)] for i in range(rows)]

K = 0.025
NO_BINS = 72 #number of bins in the wind rose
RHO = 1.225

rows = 3
cols = 3

layout_array = get_custom_layouts(SPACING)
from distributions_vC05 import wind_rose 
from AEP3_2_functions import y_5MW
turb = y_5MW()
A = turb.A
wr_list = get_wind_roses(NO_BINS)
X,Y,plot_points = rectangular_domain(EXTENT,r=RESOLUTION)

def gen_local_grid_v01C(layout,plot_points):

    xt_j,yt_j = layout[:,0],layout[:,1]
    xt_k,yt_k = plot_points[:,0],plot_points[:,1]

    x_jk = xt_k[:, None] - xt_j[None, :]
    y_jk = yt_k[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    theta_jk = np.pi/2 - np.arctan2(y_jk, x_jk)
    #this is an unneecessary conversion (!)

    return r_jk,theta_jk

theta_i = np.linspace(0,2*np.pi,NO_BINS,endpoint=False)

ff_pce_array = empty2dPyarray(rows,cols) #2d python array
pow_a_arr = empty2dPyarray(rows,cols) 
pow_b_arr = empty2dPyarray(rows,cols) 
a_ff_array = empty2dPyarray(rows,cols) 
b_ff_array = empty2dPyarray(rows,cols)

layout_array = empty2dPyarray(rows,cols)


#%%
BINS = 72
INVALID_R = None
wr1 = wind_rose(bin_no_bins=BINS,custom=4,site=None,filepath=None,a_0=10,Cp_f=turb.Cp_f) #for each wind rose
wr2 = wind_rose(bin_no_bins=BINS,custom=None,site=6,filepath=None,a_0=10,Cp_f=turb.Cp_f) #for each wind rose
wr = wr1
U_i,P_i = wr.avMagnitude,wr.frequency
theta_i = np.linspace(0,2*np.pi,BINS,endpoint=False)
layout = np.array(((0,0),),)
r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
Cp_f = turb.Cp_f
Ct_f = turb.Ct_f
a_ff = F(invalid=INVALID_R,cross_ts=True,sml_a=False,local_Cp=True)
b_ff = F(invalid=None,cross_ts=True,sml_a=True,local_Cp=True)

import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
cf = ax.contourf(X,Y,b_ff.reshape(X.shape),50,cmap=cm.coolwarm)
ax = fig.add_subplot(projection='polar',frameon=False)
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.plot(theta_i,U_i*P_i,color='black')
ax.plot(theta_i-np.pi,U_i*P_i,color='white')
#%%
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
ax.scatter(theta_i,wr1.avMagnitude*wr1.frequency)
ax.scatter(theta_i,wr2.avMagnitude*wr2.frequency)
#%%

#generate the contourf data
def pce(exact,approx):
    return 100*(exact-approx)/exact

layout_n = [3,5,7]
if INVALID_R is not None:
    print(f'overridden invalid zone = {INVALID_R}!')

for i in range(rows):
    wr = wr_list[i] #for each wind rose
    U_i,P_i = wr.avMagnitude,wr.frequency
    Ct_f = turb.Ct_f
    MEAN_CT = np.sum(turb.Ct_f(wr.avMagnitude)*wr.frequency)
    fixed_Ct_f = lambda x: MEAN_CT
    Cp_f = turb.Cp_f
    for j in range(cols):
        #get all the variables
        layout = rectangular_layout(layout_n[j],SPACING)
        r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
        a_pow = F(invalid=INVALID_R,cross_ts=True,sml_a=False,local_Cp=True)
        b_pow = F(invalid=INVALID_R,cross_ts=True,sml_a=True,local_Cp=True)
        #then the flow field(the visualisation)
        r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
        a_ff = F(invalid=INVALID_R,cross_ts=True,sml_a=False,local_Cp=True)
        b_ff = F(invalid=INVALID_R,cross_ts=True,sml_a=True,local_Cp=True)

        layout_array[i][j] = layout
        pow_a_arr[i][j] = a_pow
        pow_b_arr[i][j] = b_pow
        a_ff_array[i][j] = a_ff.reshape(X.shape)
        b_ff_array[i][j] = b_ff.reshape(X.shape)
        print("b_pow: {}".format(b_pow))
        print(f"{cols*i+(j+1)}/{rows*cols}",end="\r")

#%%plot the data ...
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
    ax.set_rlabel_position(60)  # Move radial labels away from plotted line
    ax.text(0, 0, text, ha='left',transform=ax.transAxes,color='black')
    ax.spines['polar'].set_visible(False)
    return None

from matplotlib import cm
def nice_composite_plot_v02(fig,gs,row,col,Z1,X,Y,Z2,xt,yt):
    ax = fig.add_subplot(gs[row,col])

    xticks = ax.xaxis.get_major_ticks()
    xticks[2].set_visible(False)
    ax.set_xlabel('$x/d_0$',labelpad=-9)

    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-19)
    #contourf
    cf = ax.contourf(X,Y,Z1,cmap=cm.gray)
    #scatter plot
    color_list = plt.cm.coolwarm(np.linspace(0, 1, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(color_list)
    sp = ax.scatter(xt,yt,c=Z2,cmap=cmap,marker='x',s=10,vmin=0)
    cax = fig.add_subplot(gs[row+1,col])
    cb = fig.colorbar(sp, cax=cax, orientation='horizontal',format='%.3g')
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set(xlim=(-EXTENT, EXTENT), ylim=(-EXTENT, EXTENT))
    # cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')

    if row == 4 and col == 2:
        cax.set_xlabel('Percentage Error in AEP / \%',labelpad=2)
        
    return cf

def ill_cb():
    cax = fig.add_subplot(gs[7,1:])
    cax.invert_xaxis()
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
    from matplotlib.ticker import FixedLocator, FixedFormatter
    cb.ax.xaxis.set_major_locator(FixedLocator([480, 160]))  # Set tick positions
    cb.ax.xaxis.set_major_formatter(FixedFormatter(['less waked', 'more waked']))  # Set tick labels
    #the illustrative colourbar
    return None


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(8, 4, height_ratios=[14,1,14,1,14,1,1,1],wspace=0.3,hspace=0.41)
fig = plt.figure(figsize=(7.8,8), dpi=300)

import numpy as np

for i in range(rows): 
    #first column is the wind rose
    wr = wr_list[i]
    wind_rose_y = wr.frequency*wr.avMagnitude
    nice_polar_plot(fig,gs[2*i,0],np.deg2rad(wr.deg_bin_centers),wind_rose_y,text="$P(\\theta)U(\\theta)$")
    for j in range(cols): #then onto the contours
        Z2 = pce(pow_a_arr[i][j], pow_b_arr[i][j])
        xt,yt = layout_array[i][j][:,0],layout_array[i][j][:,1]
        cf = nice_composite_plot_v02(fig,gs,2*i,j+1,a_ff_array[i][j],X,Y,Z2,xt,yt) 

ill_cb() #'illustrative' colourbar on bottom row


if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')

    print("figure saved")
