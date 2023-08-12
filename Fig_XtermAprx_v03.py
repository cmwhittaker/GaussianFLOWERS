#% A figure to show the effect of neglecting the cross terms in the product of the sums
#%% get data to plot
%load_ext autoreload
%autoreload 2
import numpy as np
SAVE_FIG = True
SPACING = 7
INVALID_R = None
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
    layout_array = [[0 for j in range(COLS)] for i in range(ROWS)]
    layout_array[:][0] = rectangular_layout(3,s)
    layout_array[:][1] = rectangular_layout(5,s)
    layout_array[:][2] = rectangular_layout(7,s)

    return layout_array

def get_wind_roses(no_bins):
    #hard-coded wind roses
    wr_list = []
    for i in range(ROWS): #for each row
        if i == 0: #the uniform wind rose
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=2,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        if i == 1: #site 6 has one strong direction 
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=3,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        if i == 2: #impulse wind rose (most extreme)
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=6,filepath=None,a_0=10,Cp_f=turb.Cp_f)
        wr_list.append(wr)
    return wr_list


def empty2dPyarray(rows,cols): #create empty 2d python array
    return [[0 for j in range(cols)] for i in range(rows)]

K = 0.03
NO_BINS = 72 #number of bins in the wind rose

ROWS = 3
COLS = 3

layout_array = get_custom_layouts(SPACING)
from distributions_vC05 import wind_rose 
from AEP3_2_functions import y_5MW
turb = y_5MW()
wr_list = get_wind_roses(NO_BINS)
X,Y,plot_points = rectangular_domain(EXTENT,r=RESOLUTION)
num_points = plot_points.shape[0]
theta_i = np.linspace(0,2*np.pi,NO_BINS,endpoint=False)
layout_n = [3,5,7]
max_num_turbs = layout_n[-1]

aep_a,aep_b = 2*[np.zeros((ROWS,COLS,num_points)),] #2d python array
nc_pow_arr = empty2dPyarray(ROWS,COLS) 
c_ff_array = empty2dPyarray(ROWS,COLS) 
nc_ff_array = empty2dPyarray(ROWS,COLS)

layout_array = empty2dPyarray(ROWS,COLS)

#generate the contourf data
def pce(exact,approx):
    return 100*(exact-approx)/exact



for i in range(ROWS):
    for j in range(COLS):
        #get all the variables
        layout = rectangular_layout(layout_n[j],SPACING)
        wr = wr_list[i]
        U_i,P_i = wr.avMagnitude,wr.frequency
        #Ct_f = np.sum(turb.Ct_f(wr.avMagnitude)*wr.frequency) #uncomment for a fixed Ct
        Ct_f = turb.Ct_f
        #aep calculations first
        _,c_pow,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=True)
        _,nc_pow,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=False)
        #then the flow field(the visualisation)
        r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
        c_ff,_,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=True)
        nc_ff,_,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=False)

        layout_array[i][j] = layout

        c_pow_arr[i][j] = c_pow
        nc_pow_arr[i][j] = nc_pow
        c_ff_array[i][j] = c_ff.reshape(X.shape)
        nc_ff_array[i][j] = nc_ff.reshape(X.shape)
        ff_pce_array[i][j] = pce(c_ff,nc_ff).reshape(X.shape)
    print(f'row {i} complete')
#%%
from AEP3_3_functions import cubeAv_v4
sanity_check,_,_ = cubeAv_v4(r_jk,theta_jk,theta_i,U_i,P_i,Ct_f,turb.Cp_f,K,turb.A,rho=1.225)

fix_me,_,_ = F(r_jk,theta_jk,theta_i,U_i,P_i,Ct_f,turb.Cp_f,K,turb.A,rho=1.225,inc_ct=True)

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

for i in range(ROWS): 
    #first column is the wind rose
    wr = wr_list[i]
    wind_rose_y = wr.frequency*wr.avMagnitude
    nice_polar_plot(fig,gs[2*i,0],np.deg2rad(wr.deg_bin_centers),wind_rose_y,text="$P(\\theta)U(\\theta)$")
    for j in range(COLS): #then onto the contours
        Z2 = pce(c_pow_arr[i][j], nc_pow_arr[i][j])
        xt,yt = layout_array[i][j][:,0],layout_array[i][j][:,1]
        cf = nice_composite_plot_v02(fig,gs,2*i,j+1,c_ff_array[i][j],X,Y,Z2,xt,yt) 

ill_cb() #'illustrative' colourbar on bottom row


if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')

    print("figure saved")
        