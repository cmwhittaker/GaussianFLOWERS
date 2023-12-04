#%% Figure to show the effect reducing the number of terms in the Fourier series
#this is based on the FarmEval 

#(rows: effect of changes to the wind rose
#colums: effect of increasing the size of the farm)

#realistically, I just need to pick a single "representative" layout/site combo and then produce a figure from that.

#%% get data to plot
import sys
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2

import numpy as np
timed = True #timing toggle
#sould this cell run?
run = True
SAVE_FIG = False
if not run:
    raise ValueError('This cell takes a long time to run - are you sure you meant to run this cell?')

SPACING = 7 #turbine spacing normalised by rotor DIAMETER
U_LIM = 3 #manually override ("user limit") the invalid radius around the turbine (otherwise variable, depending on k/Ct) - 
RESOLUTION = 100 #number of x/y points in contourf meshgrid
EXTENT = 30 #total size of contourf "window" (square from -EXTENT,-EXTENT to EXTENT,EXTENT)
K = 0.03 #expansion parameter for the Gaussian model
NO_BINS = 72 #number of bins in the wind rose

theta_i = np.linspace(0,360,NO_BINS,endpoint=False) 

from utilities.turbines import iea_10MW
turb = iea_10MW()

site_n = [2,] #[2,3,6] [6,8,10] are also tricky 
layout_n = [6,] # [5,6,7] update EXTENT to increase size of window if increasing this
Nterms = np.arange(NO_BINS/2,1,-2).astype(int).tolist() #numer of Fourier terms in the truncated series #[36,]
rot = 0

ROWS = len(site_n) #number of sites
COLS = len(layout_n) #number of layout variations
LAYS = len(Nterms) #number "layers"

#generate the contourf data
from utilities.helpers import simple_Fourier_coeffs,get_floris_wind_rose,get_WAV_pp,rectangular_layout,fixed_rectangular_domain,empty2dPyarray,empty3dPyarray,adaptive_timeit

X,Y,plot_points = fixed_rectangular_domain(EXTENT,r=RESOLUTION)

layout,powj_a,powj_b,powj_c= [empty2dPyarray(ROWS, COLS) for _ in range(4)]  #2d 
time_a,time_b,time_c = [np.zeros((ROWS,COLS)) for _ in range(3)]

powj_d,powj_e = [empty3dPyarray(ROWS, COLS, LAYS) for _ in range(2)] #need to be 3d
time_d,time_e = [np.zeros((ROWS,COLS,LAYS)) for _ in range(2)]
#flow field array
Uwff_b = np.zeros((ROWS,COLS,plot_points.shape[0]))

U_i,P_i = [np.zeros((NO_BINS,len(site_n))) for _ in range(2)]

from utilities.AEP3_functions import floris_AV_timed_aep,num_Fs,vect_num_F,ntag_PA,LC_flowers_timed_aep

for i in range(ROWS): #for each wind rose (site)
    U_i[:,i],P_i[:,i] = get_floris_wind_rose(site_n[i])
    #For ntag, the fourier coeffs are found from Cp(Ui)*Pi*Ui**3
    _,FULL_Fourier_coeffs3_PA = simple_Fourier_coeffs(turb.Cp_f(U_i[:,i])*(P_i[:,i]*(U_i[:,i]**3)*len(P_i[:,i]))/(2*np.pi))
    
    wav_Ct = get_WAV_pp(U_i[:,i],P_i[:,i],turb,turb.Ct_f) #weight ct by power production

    for j in range(COLS): #for each layout
        
        layout[i][j] = rectangular_layout(layout_n[j],SPACING,rot)
        
        #floris aep (the reference)
        powj_a[i][j],time_a[i][j] = floris_AV_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,timed=timed)

        #non-vectorised numerical aep (flow field+aep)
        aep_func_b = lambda: num_Fs(U_i[:,i],P_i[:,i],np.deg2rad(theta_i),
                                      layout[i][j],plot_points,
                                      turb,K,
                                      u_lim=None,
                                      Ct_op=1, #local Ct
                                      Cp_op=1, #local Cp
                                      cross_ts=True,ex=False)
        powj_b[i][j],_,Uwff_b[i,j,:] = aep_func_b() #no timing, performance is not comparable because it's non-vectorised

        # vectorised numerical aep (aep+time)
        # this would be the performance comparison
        aep_func_c = lambda: vect_num_F(U_i[:,i],P_i[:,i],np.deg2rad(theta_i),
                                       layout[i][j],layout[i][j],
                                       turb,
                                       K,
                                       u_lim=U_LIM,
                                       Ct_op=2, #global Ct
                                       Cp_op=1, #local Cp
                                       ex=True)
        (powj_c[i][j],_),time_c[i][j] = adaptive_timeit(aep_func_c,timed=timed)

        for k in range(LAYS): #for each number of terms
            #truncate the Fourier series
            A_n,Phi_n = FULL_Fourier_coeffs3_PA
            Trunc_Fourier_coeffs_PA = A_n[:Nterms[k]+1],Phi_n[:Nterms[k]+1]
            #then perform the calculation ...
            #ntag (No cross Terms Analytical Gaussian) (aep+time)
            aep_func_d = lambda: ntag_PA(Trunc_Fourier_coeffs_PA,
                                         layout[i][j],
                                         layout[i][j],
                                         turb,
                                         K, 
                                         #(Ct_op = 3 cnst) 
                                         #(Cp_op = 2 global)    
                                         wav_Ct)
            (powj_d[i][j][k],_),time_d[i][j][k] = adaptive_timeit(aep_func_d,timed=timed)

            #flowers AEP
            powj_e[i][j][k],time_e[i][j][k] = LC_flowers_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,0.05,Nterms=Nterms[k],timed=timed)
        
            print(f"{(k+1)+j*LAYS+i*LAYS*COLS}/{ROWS*COLS*LAYS}")

#%% process the data a bit:
# the power arrays are ragged, so I have to fix them here
aep_a,aep_b,aep_c = [np.zeros((ROWS,COLS)) for _ in range(3)] 
aep_d,aep_e = [np.zeros((ROWS,COLS,LAYS)) for _ in range(2)] 
for i in range(ROWS): 
    for j in range(COLS):
        aep_a[i,j] = np.sum(powj_a[i][j])
        aep_b[i,j] = np.sum(powj_b[i][j])
        aep_c[i,j] = np.sum(powj_c[i][j])
        for k in range(LAYS): #for each of the number of terms
            aep_d[i,j,k] = np.sum(powj_d[i][j][k])
            aep_e[i,j,k] = np.sum(powj_e[i][j][k])
Nterms_arr = np.array(Nterms) #convert to numpy array

#%% I think it is clearer to just use a single site/layout combination.
#site 2 with a 6x6 layout seems representative ...
# (this will only work when rows = 1 and cols = 1)
import matplotlib.pyplot as plt
from utilities.plotting_funcs import set_latex_font
from utilities.helpers import pce
S = 15

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 2, wspace=0.2,hspace=0.2)
fig = plt.figure(figsize=(7.8,2), dpi=300) #figsize=(7.8,8)
ax1 = fig.add_subplot(gs[0,0])

set_latex_font() #try use latex font


ax1.set(xlabel='Fourier Terms',ylabel='Normalised AEP',label='Error')
import matplotlib.ticker as ticker
ax1.invert_xaxis()
ax1.xaxis.set_major_locator(ticker.FixedLocator(Nterms_arr[::2])) 

e_g = aep_d[i,j,:]/aep_a[i,j] #Gaussian error
e_j = aep_e[i,j,:]/aep_a[i,j] #Jensen error

p_g = time_a[i,j]/time_d[i,j,:] #Gaussian performace
p_j = time_a[i,j]/time_e[i,j,:] #Jensen performance

ax1.plot(Nterms_arr,e_g,c='black',marker='o')
ax1.plot(Nterms_arr,e_j,c='grey',marker='o')

ax2 = fig.add_subplot(gs[0,1])
ax2.invert_xaxis()
import matplotlib.ticker as ticker
ax2.xaxis.set_major_locator(ticker.FixedLocator(Nterms_arr[::2])) 
ax2.plot(Nterms_arr,p_g,c='black',marker='o',label='GaussianFLOWERS')
ax2.plot(Nterms_arr,p_j,c='grey',marker='o',label='JensenFLOWERS')

ax2.legend(loc='upper left')#

ax2.set(ylabel='Times faster')

if SAVE_FIG:
    site_str = ''.join(str(x) for x in site_n)
    layout_str = ''.join(str(x) for x in layout_n)    

    from pathlib import Path

    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "fig images"
    fig_name = f"Fig_FTerm_Vary_{site_str}_{layout_str}.png"
    path_plus_name = fig_dir / fig_name
    
    plt.savefig(path_plus_name, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")

plt.show()


#%% if you are running multiple sites/layouts, this should plot them all in a grid
#plot the data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, wspace=0.5,hspace=0.5)
fig = plt.figure(figsize=(10,10), dpi=200)

from utilities.helpers import pce
for i in range(ROWS):
    for j in range(COLS):
        ax = fig.add_subplot(gs[i,j])
        ax.scatter(Nterms_arr,pce(aep_a[i,j],aep_d[i,j,:]),marker='o')
        ax2 = ax.twinx()
        ax2.scatter(Nterms_arr,time_a[i,j]/time_d[i,j,:],color='black',marker='x')

#%% if you are running multiple sites/layouts, this takes the mean of every combination and plots them to a single graph.
# (the averages get skewed by the worst cases, so I decided to use a single site/layout combination to simplfy the explanation)
fig = plt.figure(figsize=(7,7), dpi=200)
ax = fig.add_subplot()
mean_aep_pce = np.mean(pce(aep_a[:,:,None],aep_d[:,:,:]),axis=(0,1))
mean_time = np.mean(time_a[:,:,None]/time_d[:,:,:],axis=(0,1))

ax.scatter(Nterms_arr,mean_aep_pce,marker='o',label='aep error')
ax2 = ax.twinx()
ax2.scatter(Nterms_arr,mean_time,color='black',marker='x',label='aep error')
ax.legend()
plt.show()


