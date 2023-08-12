#% A figure to show the effect of neglecting the cross terms in the product of the sums
#%% get data to plot
%load_ext autoreload
%autoreload 2
import numpy as np
SAVE_FIG = True
SPACING = 7
INVALID_R = None
RESOLUTION = 100
EXTENT = 25
flag = 1

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
    #hard-coded wind roses
    wr_list = []
    for i in range(rows): #for each row
        if i == 0: #the uniform wind rose
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=2,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        if i == 1: #site 6 has one strong direction 
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=3,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        if i == 2: #impulse wind rose (most extreme)
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=6,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        wr_list.append(wr)
    return wr_list

def F(r_jk,theta_jk,theta_i,U_i,P_i,ct_f,cp_f,K,A,rho=1.225,inc_ct=True):
    #function to show the difference caused by neglecting the cross terms. note: this is NOT the same as cubeAv_v4. The power coefficient is approximated using (directional dependant) global inflow.
    #i:directions,j:turbines,k:turbines in superposistion
    def deltaU_by_Uinf(r,theta,ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))

        U_delta_by_U_inf = (1-np.sqrt(1-(ct/(8*(K*r*np.sin(theta)+ep)**2))))*(np.exp(-(r*np.cos(theta))**2/(2*(K*r*np.sin(theta)+ep)**2)))

        lim = (np.sqrt(ct/8)-ep)/K #this is the y value of the invalid region, can be negative depending on Ct
        if INVALID_R is not None:
            string = f'overridden invalid zone= {INVALID_R}!'
            print(f'{flag*(string)}')
            flag = 0 
        lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be >0 but there is numerical artifacting in rsin(theta) )
        deltaU_by_Uinf = np.where(r*np.sin(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
        return deltaU_by_Uinf

    def soat(a): #Sum over Axis Two
        return np.sum(a,axis=2)
    
    #I sometimes use this function to find the wake layout, so find relative posistions to plot points not the layout 
    #when plot_points = layout it finds wake at the turbine posistions
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None] + 3*np.pi/2 # I don't know

    r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)
    ct_ijk = ct_f(U_i)[...,None,None]*np.ones((r_jk.shape[0],r_jk.shape[1]))[None,...] #this is a dirty way of repeating along 2 axis

    a = deltaU_by_Uinf(r_ijk,theta_ijk,ct_ijk,K) #deltaU_by_Uinf
    if inc_ct: #INcluding cross terms
        Uw_ij_cube = (U_i[:,None]*(1-np.sum(a,axis=2)))**3
    else: #EXcluding cross terms (soat = Sum over Axis Two)
        Uw_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(a) + 3*soat(a**2) - soat(a**3))
    #the directional-dependent thrust coefficient is approximated using global inflow speeds
    CpUinf_i = cp_f(U_i[:,None])

    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_ij = P_i[:,None]*(0.5*A*rho*CpUinf_i*Uw_ij_cube)/(1*10**6) 
        aep = np.sum(pow_ij)
        flow_field = np.nan   

    else: #farm wake visualisation
        pow_ij = np.nan
        aep = np.nan
        flow_field = np.sum(CpUinf_i*P_i[:,None]*Uw_ij_cube,axis=0)

    return flow_field,np.sum(pow_ij,axis=0),aep

def empty2dPyarray(rows,cols): #create empty 2d python array
    return [[0 for j in range(cols)] for i in range(rows)]

K = 0.03
NO_BINS = 72 #number of bins in the wind rose

rows = 3
cols = 3

layout_array = get_custom_layouts(SPACING)
from distributions_vC05 import wind_rose 
from AEP3_2_functions import y_5MW
turb = y_5MW()
wr_list = get_wind_roses(NO_BINS)
X,Y,plot_points = rectangular_domain(EXTENT,r=RESOLUTION)
from AEP3_3_functions import gen_local_grid_v01C
theta_i = np.linspace(0,2*np.pi,NO_BINS,endpoint=False)

ff_pce_array = empty2dPyarray(rows,cols) #2d python array
c_pow_arr = empty2dPyarray(rows,cols) 
nc_pow_arr = empty2dPyarray(rows,cols) 
c_ff_array = empty2dPyarray(rows,cols) 
nc_ff_array = empty2dPyarray(rows,cols)

layout_array = empty2dPyarray(rows,cols)

#generate the contourf data
def pce(exact,approx):
    return 100*(exact-approx)/exact

layout_n = [3,5,7]

for i in range(rows):
    for j in range(cols):
        #get all the variables
        layout = rectangular_layout(layout_n[j],SPACING)
        wr = wr_list[i]
        r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
        U_i,P_i = wr.avMagnitude,wr.frequency
        fixed_Ct_f = lambda x: turb.Ct_f(np.mean(wr.avMagnitude*wr.frequency*NO_BINS))
        #aep calculations first
        _,c_pow,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,fixed_Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=True)
        _,nc_pow,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,fixed_Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=False)
        #then the flow field(the visualisation)
        r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
        c_ff,_,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,fixed_Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=True)
        nc_ff,_,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,fixed_Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=False)

        layout_array[i][j] = layout

        c_pow_arr[i][j] = c_pow
        nc_pow_arr[i][j] = nc_pow
        c_ff_array[i][j] = c_ff
        nc_ff_array[i][j] = nc_ff
        ff_pce_array[i][j] = pce(c_ff,nc_ff).reshape(X.shape)
    print(f'row {i} complete')
#%%
from AEP3_3_functions import cubeAv_v4
sanity_check,_,_ = cubeAv_v4(r_jk,theta_jk,theta_i,U_i,P_i,fixed_Ct_f,turb.Cp_f,K,turb.A,rho=1.225)

fix_me,_,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,fixed_Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=True)

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
def nice_composite_plot_v02(fig,gs,row,col,X,Y,Z,xt,yt):
    ax = fig.add_subplot(gs[row,col])
    Z = np.where(Z>100,np.nan,Z) #clip Z error
    cf = ax.contourf(X,Y,Z,cmap=cm.coolwarm)
    #colourbar and decorations
    cax = fig.add_subplot(gs[row+1,col])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
    cb.ax.locator_params(nbins=6)
    cb.ax.tick_params(labelsize=9)
    ax.scatter(xt,yt,marker='x',color='black',s=10)
    return None

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(6, 4, height_ratios=[14,1,14,1,14,1],wspace=0.3,hspace=0.37)
fig = plt.figure(figsize=(7.8,7), dpi=100)

import numpy as np

for i in range(rows): 
    #first column is the wind rose
    wr = wr_list[i]
    wind_rose_y = wr.frequency*wr.avMagnitude
    nice_polar_plot(fig,gs[2*i,0],np.deg2rad(wr.deg_bin_centers),wind_rose_y,text="$P(\\theta)U(\\theta)$")
    for j in range(cols): #then onto the contours
        xt,yt = layout_array[i][j][:,0],layout_array[i][j][:,1]
        nice_composite_plot_v02(fig,gs,2*i,j+1,X,Y,ff_pce_array[i][j],yt,xt) 
        
if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches="tight")
        
#%% sanity check
xi,yi = 2,2
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
#Z = ff_pce_array[xi][yi].reshape(X.shape) 
Z = nc_ff_array[xi][yi].reshape(X.shape)
#Z = np.where(np.abs(Z)>10,0,Z)
#Z = c_ff.reshape(X.shape)

cf = ax.contourf(Y,X,Z,50,cmap=cm.coolwarm)
fig.colorbar(cf)
ax.scatter(layout_array[xi][yi][:,1],layout_array[xi][yi][:,0])
