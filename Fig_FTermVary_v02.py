#%% Figure to show the effect reducing the number of terms in the Fourier series

#this is based on Fig_FarmEval v03
#%% get data to plot
%load_ext autoreload
%autoreload 2
import numpy as np
SAVE_FIG = False
SPACING = 7
INVALID_R = None
RESOLUTION = 100
EXTENT = 30
flag = 1

def rectangular_domain(extent,r=200):
    X,Y = np.meshgrid(np.linspace(-extent,extent,r),np.linspace(-extent,extent,r))
    return X,Y,np.column_stack((X.reshape(-1),Y.reshape(-1)))

def rectangular_layout(no_xt,s,rot):
    low = (no_xt)/2-0.5
    xt = np.arange(-low,low+1,1)*s
    yt = np.arange(-low,low+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)
    Xt,Yt = [_.reshape(-1) for _ in [Xt,Yt]]
    rot_Xt = Xt * np.cos(rot) + Yt * np.sin(rot)
    rot_Yt = -Xt * np.sin(rot) + Yt * np.cos(rot) 
    layout = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
    return layout#just a single layout for now

def empty2dPyarray(rows,cols): #create empty 2d python array
    return [[0 for j in range(cols)] for i in range(rows)]

def empty3dPyarray(rows,cols,lays): #create empty 2d python array
    return [[[0 for k in range(lays)] for j in range(cols)] for i in range(rows)]

from floris.tools import WindRose
def get_floris_wind_rose(site_n):
    fl_wr = WindRose()
    folder_name = "WindRoseData_D/site" +str(site_n)
    fl_wr.parse_wind_toolkit_folder(folder_name,limit_month=None)
    wr = fl_wr.resample_average_ws_by_wd(fl_wr.df)
    wr.freq_val = wr.freq_val/np.sum(wr.freq_val)
    U_i = wr.ws
    P_i = wr.freq_val
    return np.array(U_i),np.array(P_i)

import time
def floris_timed_aep(U_i,P_i,theta_i,layout,turb,wake=True,timed=True):
    from floris.tools import FlorisInterface
    fi = FlorisInterface("floris_settings.yaml")
    fi.reinitialize(wind_directions=theta_i, wind_speeds=U_i, time_series=True,layout_x=turb.D*layout[:,0],layout_y=turb.D*layout[:,1])
    if wake:
        if timed:
            timings = %timeit -o -q fi.calculate_wake()
            time = timings.best
        else:
            fi.calculate_wake()
            time = np.NaN
    else:
        fi.calculate_no_wake()
        time = np.NaN
    aep_array = fi.get_turbine_powers() #happy days, we can go from here
    pow_j = np.sum(P_i[:,None,None]*aep_array,axis=0)
    return pow_j/(1*10**6),time

def analytical_timed_aep(Fourier_coeffs_PA,layout,WAV_CT,K,turb,timed=True):
    from AEP3_3_functions import ntag_PA_v03
    if timed:
        result = []
        timings = %timeit -o -q result.append(ntag_PA_v03(Fourier_coeffs_PA,layout,layout,turb,WAV_CT,K))
        pow_j,_ = result[0]
        time = timings.best
    else:
        pow_j,_ = ntag_PA_v03(Fourier_coeffs_PA,layout,layout,turb,WAV_CT,K)
        time = np.NaN
    return pow_j,time

def numerical_aep(U_i,P_i,theta_i,layout,plot_points,turb,K):
    WAV_CP = np.sum(turb.Cp_f(U_i)*P_i)
    pow_j,_,Uwff_ja= num_F_v02(U_i,P_i,theta_i,
                       layout,
                       plot_points,
                       turb,
                       K=K,
                       u_lim=U_LIM,
                       Ct_op=2,Cp_op=1,cross_ts=True,ex=True)
    return pow_j,Uwff_ja

run = False
if not run:
    raise ValueError('Did you fat finger this cell?')

K = 0.03
U_LIM = None
NO_BINS = 72 #number of bins in the wind rose
theta_i = np.linspace(0,360,NO_BINS,endpoint=False)

ROWS = 3
COLS = 3


from turbines_v01 import iea_10MW
turb = iea_10MW()

site_n = [2,3,6] #[2,3,6]
layout_n = [5,6,7]
rot = [0,0,0]
Nterms = np.arange(NO_BINS/2,0,-6).astype(int).tolist() #+ [3,2,1,0]
LAYS = len(Nterms)

X,Y,plot_points = rectangular_domain(EXTENT,r=RESOLUTION)

Uwff_c = np.zeros((ROWS,COLS,plot_points.shape[0]))

layout = empty2dPyarray(ROWS, COLS) 
powj_a,time_a,powj_b,time_b,powj_c,time_c,powj_d = [empty3dPyarray(ROWS, COLS, LAYS) for _ in range(7)]  #3d python array!

U_i,P_i = [np.zeros((NO_BINS,len(site_n))) for _ in range(2)]

#generate the contourf data
from AEP3_3_functions import num_F_v02,simple_Fourier_coeffs_v01
for i in range(ROWS): #for each wind rose
    U_i[:,i],P_i[:,i] = get_floris_wind_rose(site_n[i])
    _,FULL_Fourier_coeffs_PA = simple_Fourier_coeffs_v01(turb.Cp_f(U_i[:,i])*(P_i[:,i]*(U_i[:,i]**3)*len(P_i[:,i]))/(2*np.pi))
    WAV_CT = np.sum(turb.Ct_f(U_i[:,i])*P_i[:,i])

    for j in range(COLS): #for each layout
        layout[i][j] = rectangular_layout(layout_n[j],SPACING,rot[j])

        #floris aep (the reference)
        powj_a[i][j],time_a[i][j] = floris_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,timed=True)

        #numerical aep (for flow field and as a reference)
        powj_c[i][j],Uwff_c[i,j,:] = numerical_aep(U_i[:,i],P_i[:,i],np.deg2rad(theta_i),layout[i][j],plot_points,turb,K)

        for k in range(LAYS): #for each number of terms  
            a_0,A_n,Phi_n = FULL_Fourier_coeffs_PA #truncate Fourier
            Fourier_coeffs_PA = a_0, A_n[:Nterms[k]],Phi_n[:Nterms[k]]

            #analytical aep
            powj_b[i][j][k],time_b[i][j][k] = analytical_timed_aep(Fourier_coeffs_PA,layout[i][j],WAV_CT,K,turb,timed=True)
            
            print(f"{(k+1)+j*LAYS+i*LAYS*COLS}/{ROWS*COLS*LAYS}",end="\r")
     
#%% now plot the results
import matplotlib.pyplot as plt
from AEP3_3_functions import pce
#the power arrays are ragged, so I have to fix them here
error_arr = np.zeros((ROWS,COLS,LAYS))
aep_a = np.zeros((ROWS,COLS))
aep_b = np.zeros((ROWS,COLS,LAYS))
for i in range(ROWS): 
    for j in range(COLS):
        aep_a[i,j] = np.sum(powj_a[i][j][0])
        for k in range(LAYS):
            aep_b[i,j,k] = np.sum(powj_b[i][j][k])
time_a = np.array(time_a)
time_b = np.array(time_b) #fix this also
Nterms_arr = np.array(Nterms)

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

def nice_plot():
    #this is using globals ...
    S = 15
    fig,ax = plt.subplots(figsize=(3,3),dpi=300)
    
    s1 = ax.scatter(x3,y3,c='black',marker='o',s=S)
    ax.set(xlabel='Fourier Terms',ylabel='Error Magnification',label='Error')
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.FixedLocator(Nterms)) 
    
    ax2 = ax.twinx()
    s2 = ax2.scatter(x3,y4,c='black',marker='x',s=S,label='Performance')
    ax2.set(ylabel='Performance Magnification')

    ax.legend([s1, s2], ["Error", "Performance"])

    return None
y1 = pce(aep_a[:,:,None],aep_b) #errors compared to ref
y2 = y1 / y1[:,:,0:1] #errors compared to full term
y3 = np.mean(y2,axis=(0,1)) #mean error over every rose/layout
x3 = Nterms

y4 = np.mean(time_a[:,:,None]/time_b,axis=(0,1))
y4 = y4/y4[0] #normalise by the full term
nice_plot()

if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')

    print("figure saved")

#%% The following cells (1/3) are useful for explaining the plot
# error of each farm, each layout and each number of terms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, wspace=0.5,hspace=0.5)
fig = plt.figure(figsize=(10,10), dpi=200)
for i in range(ROWS): 
    for j in range(COLS):
        ax = fig.add_subplot(gs[i,j])
        ax.scatter(Nterms_arr[None,None,:],pce(aep_a[i,j,None],aep_b[i,j,None]),color='black')
        ax2 = ax.twinx()
        ax2.scatter(Nterms_arr[None,None,:],time_a[i,j,None]/time_b[i,j,:],color='black',marker='x')

fig.legend
#%% The following cells (2/3) are useful for explaining the plot
# plot all that onto one (error only)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
col_list = np.array(['blue','orange','red'])
col_list = np.broadcast_to(col_list[:,None,None],error_arr.shape)
#ax.scatter(Nterms_arr,error_arr,c=col_list.flatten())
y1 = pce(aep_a[:,:,None],aep_b)
x1 = np.broadcast_to(Nterms_arr[None,None,:],y1.shape)
ax.scatter(x1,y1,c=col_list.flatten())
#%% The following cells (3/3) are useful for explaining the plot
# then normalise by the full term error
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
col_list = np.array(['blue','orange','red'])
col_list = np.broadcast_to(col_list[:,None,None],error_arr.shape)
#ax.scatter(Nterms_arr,error_arr,c=col_list.flatten())
y2 = y1 / y1[:,:,0:1]
x2 = np.broadcast_to(Nterms_arr[None,None,:],y1.shape)
ax.scatter(x2,y2,c=col_list.flatten())

#%%
print("hello world")

