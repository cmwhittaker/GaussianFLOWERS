#%% find aep using the different methods for a random layout

#%% get data to plot
import sys
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2

import numpy as np

run = 1
SAVE_FIG = False
timed = False 
if not run:
    response = input('This cell takes a long time to run - are you sure you meant to run this cell? Press any key to continue or type "n" to cancel: ')
    if response.lower() == 'n':
        raise ValueError('Execution cancelled by the user.')
    

U_LIM = 3 #manually override ("user limit") the invalid radius around the turbine (otherwise variable, depending on k/Ct) - 
RESOLUTION = 100 #number of x/y points in contourf meshgrid
EXTENT = 35 #total size of contourf "window" (square from -EXTENT,-EXTENT to EXTENT,EXTENT)
K = 0.03 #expansion parameter for the Gaussian model
Kj = 0.05 #expansion parameter for the Jensen model
NO_BINS = 360 #number of bins in the wind rose
ALIGN_WEST = True
SYNTHETIC_WR = False

site_var = [1,2,3,4,5,6,7,8,9,10,11,12] # [1,2,3,4,5,6,7,8,9,10,11,12] # [2,1,8]:[30,40,25] 

# from utilities.helpers import random_layouts
# NO_LAYOUTS = 4
# np.random.seed(1)
# layouts = random_layouts(NO_LAYOUTS)

layouts = np.load('rdm_layouts.npy', allow_pickle=True)
NO_LAYOUTS = len(layouts)

ROWS = len(site_var) #number of sites
COLS = len(layouts)

if not run: #I used ipy and don't want to fat finger and wait 20 min for it to run again
    raise ValueError('This cell takes a long time to run - are you sure you meant to run this cell?')

def find_errors(U_i,P_i,theta_i,plot_points,layout,turb,K):
    # this finds the errors resulting from each of the assumptions, they are:
    # 1. Ct_error: Approximating Ct(U_w) (local) with a constant \overline{C_t}
    # 2. Cp_error1: Approximating Cp(U_w) (local) with Cp(U_\infty) (global)
    # 3. Cx1_error: Cros terms approximation Approximating ( \sum x )^n with ( \sum x^n )
    # 4. SA_error: small angle approximation of the Gaussian wake model (sin(\theta) \approx \theta etc...)    

    #WAV_Ct shouldn't really be global
    from utilities.AEP3_functions import num_Fs
    from utilities.helpers import pce
    
    def simple_aep(Ct_op=1,Cp_op=1,cross_ts=True,ex=True,cube_term=True):
        pow_j,_,_= num_Fs(U_i,P_i,theta_i,
                     plot_points,layout,
                     turb,
                     K=K,
                     u_lim=None,
                     Ct_op=Ct_op,wav_Ct=wav_Ct,
                     Cp_op=Cp_op,wav_Cp=None,
                     cross_ts=cross_ts,ex=ex,cube_term=cube_term)
        return np.sum(pow_j)
    exact = simple_aep() #the default options represent no assumptions takene
    Ct_error = pce(exact,simple_aep(Ct_op=3)) #Ct_op 3 is a constant Ct
    Cp_error1 = pce(exact,simple_aep(Cp_op=2)) #Cp_op 2 is a global Cp
    Cx1_error = pce(exact,simple_aep(cross_ts=False)) #neglect cross terms
    SA_error = pce(exact,simple_aep(ex=False)) #ex:"exact" =False so use small angle approximation
    
    return (Ct_error,Cp_error1,Cx1_error,SA_error)

import warnings
# Suppress spammy runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

thetaD_i = np.linspace(0,360,NO_BINS,endpoint=False) #theta in degrees
thetaD_WB_i = 270 - thetaD_i #wind bearing bin centers 

from utilities.turbines import iea_10MW
turb = iea_10MW()

#generate the contourf data
from utilities.helpers import empty2dPyarray,simple_Fourier_coeffs,get_floris_wind_rose,get_WAV_pp,rectangular_layout,fixed_rectangular_domain,adaptive_timeit,vonMises_wr

X,Y,plot_points = fixed_rectangular_domain(EXTENT,r=RESOLUTION)

layout,powj_a,powj_b,powj_c,powj_d,powj_e,powj_f,powj_g,powj_h= [empty2dPyarray(ROWS, COLS) for _ in range(9)]  #2d python arrays
time_a,time_b,time_c,time_d,time_e,time_g,time_h = [np.zeros((ROWS,COLS)) for _ in range(7)]

Uwff = np.zeros((ROWS,COLS,plot_points.shape[0]))

U_i,P_i = [np.zeros((NO_BINS,len(site_var))) for _ in range(2)]

errors = np.zeros((ROWS,COLS,4))

from utilities.AEP3_functions import floris_AV_timed_aep,num_Fs,vect_num_F,ntag_PA,caag_PA,floris_FULL_timed_aep,jflowers

for i in range(ROWS): #for each wind rose (site)
    U_i[:,i],P_i[:,i],_,fl_wr = get_floris_wind_rose(site_var[i],align_west=ALIGN_WEST,n_bins=NO_BINS)
    #For ntag, the fourier coeffs are found from Cp(Ui)*Pi*Ui**3
    _,Fourier_coeffs3_PA = simple_Fourier_coeffs(turb.Cp_f(U_i[:,i])*(P_i[:,i]*(U_i[:,i]**3)*len(P_i[:,i]))/(2*np.pi))
    #For caag, the fourier coeffs are found from Pi*Ui
    _,Fourier_coeffs_noCp_PA = simple_Fourier_coeffs((P_i[:,i]*U_i[:,i]*len(P_i[:,i]))/(2*np.pi))
    #weight ct by power production
    wav_Ct = get_WAV_pp(U_i[:,i],P_i[:,i],turb,turb.Ct_f) 

    #for Jensen FLOWERS, the Fourier coeffs are found from
    # 1-sqrt(ct) etc.
    c_0 = np.sum(U_i[:,i]*P_i[:,i])/np.pi
    Fourier_coeffs_j,_ = simple_Fourier_coeffs((1 - np.sqrt(1 - turb.Ct_f(U_i[:,i]))) * U_i[:,i]*P_i[:,i]*len(P_i[:,i])/(2*np.pi)) 

    for j in range(COLS): 
        layout = layouts[j]
        #find the errors due to each assumption (numerically)
        errors[i,j,:] = find_errors(U_i[:,i],P_i[:,i],np.deg2rad(thetaD_i),plot_points,layout,turb,K)

        #floris aep (the reference)
        powj_a[i][j],time_a[i][j] = floris_AV_timed_aep(U_i[:,i],P_i[:,i],thetaD_WB_i,layout,turb,timed=timed)

        #non-vectorised numerical aep (flow field+aep)
        aep_func_b = lambda: num_Fs(U_i[:,i],P_i[:,i],np.deg2rad(thetaD_i),
                                    plot_points,layout,
                                    turb,K,
                                    u_lim=None,
                                    Ct_op=1, #local Ct
                                    Cp_op=1, #local Cp
                                    cross_ts=True,ex=True,
                                    ff=False)
        (powj_b[i][j],_,_),time_b[i][j] = adaptive_timeit(aep_func_b,timed=timed)

        #flow field (for visualisation - no timing) using num_F
        _,_,Uwff[i,j,:] = num_Fs(U_i[:,i],P_i[:,i],np.deg2rad(thetaD_i),
                                    plot_points,layout,
                                    turb,K,
                                    u_lim=None,
                                    Ct_op=1, #local Ct
                                    Cp_op=1, #local Cp
                                    cross_ts=True,ex=True,
                                    ff=True)


        #vectorised numerical aep (aep+time)
        aep_func_c = lambda: vect_num_F(U_i[:,i],P_i[:,i],
                                       np.deg2rad(thetaD_i),
                                       layout,layout,
                                       turb,
                                       K,
                                       u_lim=U_LIM,
                                       Ct_op=2, #global Ct
                                       Cp_op=1, #local Cp
                                       ex=True)
        (powj_c[i][j],_),time_c[i][j] = adaptive_timeit(aep_func_c,timed=timed)

        #ntag (No cross Terms Analytical Gaussian) (aep+time)
        aep_func_d = lambda: ntag_PA(Fourier_coeffs3_PA,
                                         layout,
                                         layout,
                                         turb,
                                         K, 
                                         #(Ct_op = 3 cnst) 
                                         #(Cp_op = 2 global )    
                                         wav_Ct)
        (powj_d[i][j],_),time_d[i][j] = adaptive_timeit(aep_func_d,timed=timed)

        #caag (cube of the average) analytical aep
        aep_func_e = lambda: caag_PA(Fourier_coeffs_noCp_PA,
                                         layout,
                                         layout,
                                         turb,
                                         K,
                                         #(Ct_op = 3 cnst) 
                                         #(Cp_op = 2 *local-ish)
                                         wav_Ct)
        # *local based on the weight averaged wake velocity 
        (powj_e[i][j],_),time_e[i][j] = adaptive_timeit(aep_func_e,timed=timed)

        # #floris NO WAKE aep (sanity check)
        powj_f[i][j],_ = floris_AV_timed_aep(U_i[:,i],P_i[:,i],thetaD_WB_i,layout,turb,wake=False,timed=False)   

        #flowers AEP
        aep_func_g = lambda: jflowers(Fourier_coeffs_j,
                                      layout,layout,
                                      turb,
                                      Kj,
                                      c_0,
                                      RHO=1.225,
                                      r_lim=0.5)
        (powj_g[i][j],_),time_g[i][j] = adaptive_timeit(aep_func_g,timed=timed)
        
        #floris AEP WITHOUT wind speed averaging
        # powj_h[i][j],time_h[i][j] = floris_FULL_timed_aep(fl_wr,layout,turb,timed=False)

        print(f"{COLS*i+(j+1)}/{ROWS*COLS}\r")

#%%     
#convert ragged arrays and format other meta data
        
from sklearn.metrics.pairwise import euclidean_distances
def mnn_f(layout):
    #mean nearest neighbour
    distances = euclidean_distances(layout,layout)
    distances[distances<0.1]=np.nan
    mnn= np.mean(np.nanmin(distances,axis=1))
    return mnn

powjs = [powj_a,powj_b,powj_c,powj_d,powj_g]
aep_arr = np.zeros((5,ROWS,COLS))
n_turb_arr = np.zeros((ROWS,COLS))
mean_ws_arr = np.zeros((ROWS,COLS))
mnn_arr = np.zeros((ROWS,COLS))
for i in range(ROWS):
    for j in range(COLS):
        n_turb_arr[i,j] = len(layouts[j])
        mnn_arr[i,j] = mnn_f(layouts[j])
        mean_ws_arr[i,j] = np.sum(U_i[:,i]*P_i[:,i])
        for a in range(5): #aep is a ""special" list
            aep_arr[a,i,j] = np.sum(powjs[a][i][j])

aep_arr_n = aep_arr/aep_arr[0,:,:] #normalise by cumCurl
aep_arr_n = aep_arr_n.reshape(5,ROWS*COLS)
n_turb_arr = n_turb_arr.reshape(ROWS*COLS)
#mean_ws_arr = mean_ws_arr.reshape(ROWS*COLS)
mnn_arr = mnn_arr.reshape(ROWS*COLS)

#%% boxplot of normalised AEP
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)
#%%
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(3,1.5),dpi=300)

boxplot = a = ax.violinplot(aep_arr_n.swapaxes(0, 1))

boxplot = a = ax.boxplot(aep_arr_n.swapaxes(0, 1)[:,1:5],showfliers=False,vert=False)
median_lines = boxplot['medians']
for line in median_lines:
    line.set_color('black')
labels = ['Numerical Integration', 'Vectorised Numerical \n Integration', 'Gaussian-Flowers', 'Jensen-Flowers']
# Set custom labels for the x-axis points
ax.set_yticks(range(1, len(labels) + 1))
_ = ax.set_yticklabels(labels, rotation='horizontal')
ax.set(xlabel='Normalised AEP')
ax.invert_yaxis()


LW = 0.5
# Set the line width of the boxes
for box in boxplot['boxes']:
    # set edge color and line width
    box.set(linewidth=LW)

# Set the line width of the whiskers
for whisker in boxplot['whiskers']:
    whisker.set(linewidth=LW)

# Set the line width of the caps
for cap in boxplot['caps']:
    cap.set(linewidth=LW)

# Set the line width of the medians
for median_err in boxplot['medians']:
    median_err.set(linewidth=LW)

#%% the trends 3 column figure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from utilities.plotting_funcs import set_latex_font
gs = GridSpec(1,4,width_ratios=(1,1,1,0.4),wspace=0.1,hspace=0)
fig = plt.figure(figsize=(7.8,2), dpi=300) 

ft = 1 #first time flag

cmap = plt.cm.viridis
colors = [cmap(0.0), cmap(0.5)] 
set_latex_font()


#first is
def lin_reg_scatter(ax,x):
    global ft
    sort_indx = np.argsort(x)
    xs = x[sort_indx] #x sortef

    y1 = aep_arr_n[3,:][sort_indx] #Gaussian AEP
    y2 = aep_arr_n[4,:][sort_indx] #Jensen AEP

    ax.scatter(xs,y1,alpha=0.5,s=15,marker='x',label=ft*"GF Points",color=colors[0])
    # Linear regression 1 (Gaussian Flowers)
    m, b = np.polyfit(xs,y1,1)
    ax.plot(xs, m*xs + b, lw=1,label=ft*"GF Linear Fit",color=colors[0])

    ax.scatter(xs,y2,alpha=0.5,s=15,marker='+',label=ft*"JF Points",color=colors[1])
    #Linear regression 2 (Jensen Flowers)
    m, b = np.polyfit(xs,y2,1)
    ax.plot(xs, m*xs + b,lw=1,ls='-',label=ft*"JF Linear Fit",color=colors[1])
    
    
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ft = 0


ax1 = fig.add_subplot(gs[0])
lin_reg_scatter(ax1,mean_ws_arr.reshape(-1))
ax1.set(xlabel="Mean Wind Speed / $ms^{-1}$",ylabel="Normalised AEP")

ax2 = fig.add_subplot(gs[1])
lin_reg_scatter(ax2,n_turb_arr)
ax2.set(xlabel="No. Turbines",yticklabels=[])

ax3 = fig.add_subplot(gs[2])
ax3.invert_xaxis()
lin_reg_scatter(ax3,mnn_arr)
ax3.set(xlabel="Turbine Spacing / $D$",yticklabels=[])

#ax4 = fig.add_subplot(gs[3])
fig.legend(ncols=1,borderaxespad=1.32)

#%% get the data to put in the table
for i in range(len(labels)):
    data = aep_arr_n[i+1,:]
    # Median
    median_err = (np.median(data)-1)*100
    # IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    std_dev = 100*np.std(data,ddof=1) 

    print(f"{labels[i]}- Err: {median_err:.2f}, IQR:{IQR:.1f}, std: {std_dev:.2f} ")

#%% layout reference
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
gs = GridSpec(5, 5,wspace=0.3)
fig = plt.figure(figsize=(8,8), dpi=200) 

for i in range(NO_LAYOUTS):
    ax = fig.add_subplot(gs[i])
    ax.set(aspect='equal')
    layout = layouts[i]
    ax.scatter(layout[:,0],layout[:,1],marker='o',color='black',s=3,lw=1)
    ax.set(xlim=(-EXTENT,EXTENT),ylim=(-EXTENT,EXTENT))
    xticks = ax.xaxis.get_major_ticks()
    xticks[2].set_visible(False)
    ax.set_xlabel('$x/D$',labelpad=-10)

    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/D$',labelpad=-15)
    ax.yaxis.set_tick_params(pad=0)


#%%
n_turbs =[]
for i in range(NO_LAYOUTS):
    n_turbs.append(len(layouts[i]))

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,3),dpi=200)
ax.hist(n_turbs)

#%%
b = []
for j in range(3):
    for i in range(3):
        b.append(8.760*np.sum(powj_g[i][j]))