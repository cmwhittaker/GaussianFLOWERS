#%% A mega figure to compare the performance of the original and the "better" method
#Starting with a single row of the figure that i can then paste to create a massive one
%load_ext autoreload
%autoreload 2

K = 0.025
square_no = [2,2,3] #number of turbines on one side of the farm
SPACING = 6
sites = [10,10,10] #3 for now (site 6 raises errors!)
NO_BINS = 72
RESOLUTION = 50
SAVE_FIG = False

ti = index = next((i for i, num in enumerate(square_no) if num > 9), -1)

import numpy as np
def rectangular_layout(no_xt,no_yt,s=5):
    xt = np.arange(1,no_xt+1,1)*s
    yt = np.arange(1,no_yt+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)
    return Xt.reshape(-1),Yt.reshape(-1),np.column_stack((Xt.reshape(-1),Yt.reshape(-1))), np.size(Xt)#just a single layout for now

def random_layout(sl,spacing=5):
    from AEP3_functions_v01 import poisson2dRandomPoints
    np.random.seed(12341234)
    layouts,distances = poisson2dRandomPoints(1,sl,sl,min_spacing=spacing,k=30)
    mean_dist = np.mean(distances)
    layout = layouts[0,:,:]
    layout = layout[~np.isnan(layout).any(axis=1)]
    return layout[:,0],layout[:,1],layout,layout[:,0].size,mean_dist

def rectangular_domain(layout,s=5,pad=4,r=200):
    Xt,Yt = layout[:,0],layout[:,1]
    pad = 1.0
    xr,yr = r,r #resolution
    X,Y = np.meshgrid(np.linspace(np.min(Xt)-pad*s,np.max(Xt)+pad*s,xr),np.linspace(np.min(Yt)-pad*s,np.max(Yt)+pad*s,yr))
    return X,Y,np.column_stack((X.reshape(-1),Y.reshape(-1)))

#setup spam
from turbines_v01 import iea_10MW
turb = iea_10MW()
theta_i = np.linspace(0,2*np.pi,NO_BINS,endpoint=False)
from AEP3_3_functions import gen_local_grid_v01C,cubeAv_v4,ntag_v02,ca_ag_v02

from distributions_vC05 import wind_rose
def get_own_wind_rose(site_n):
    #my own wind rose 
    own_wr = wind_rose(bin_no_bins=NO_BINS,custom=None,a_0=8,site=site_n,Cp_f=turb.Cp_f)
    #wind rose for floris
    MEAN_CT = np.sum(turb.Ct_f(own_wr.avMagnitude)*own_wr.frequency)
    return own_wr,MEAN_CT

from floris.tools import FlorisInterface, WindRose
fls_wr = WindRose()

def initalise_floris_wind_rose(site_n):
    folder_name = "WindRoseData_D/site" +str(site_n)
    fls_wr.parse_wind_toolkit_folder(folder_name,limit_month=None)
    return None

import time
def timed_floris_aep():
    start_time = time.time()
    fi.reinitialize(layout_x=turb.D*layout[:,0],layout_y=turb.D*layout[:,1])
    aep = fi.get_farm_AEP_wind_rose_class(wind_rose=fls_wr)
    ellapsed_time = time.time()-start_time
    return aep/((1*10**6)*365* 24),ellapsed_time

def timed_floris_aep2(wake=True):
    # a lil function to make the above more readable
    wr_speed = own_wr.avMagnitude
    wr_freq = own_wr.frequency
    no_bins = wr_speed.size
    theta_i = np.linspace(0,360,no_bins,endpoint=False)
    Nt = layout.shape[0] #more readable
    pow_ij = np.zeros((no_bins,Nt))
    fi.reinitialize(layout_x=turb.D*layout[:,0],layout_y=turb.D*layout[:,1])
    start_time = time.time()
    for i in range(no_bins): #for each bin
        fi.reinitialize(
        #pretty silly
        wind_directions=np.array((theta_i[i],)),
        wind_speeds=np.array((wr_speed[i],)) #this will include the frequency
        )
        if wake == True:
            fi.calculate_wake() 
        else:
            fi.calculate_no_wake()
        
        pow_ij[i,:] = wr_freq[i]*fi.get_turbine_powers()/(1*10**6)

    ellapsed_time = time.time()-start_time

    pow_j = np.sum(pow_ij,axis=0)
    aep = np.sum(pow_ij)
    return pow_j,aep,ellapsed_time

def timed_numerical_aep():
    #this is a botch, but it does work!
    result = []
    timings = %timeit -o -q result.append(cubeAv_v4(r_jk,theta_jk,theta_i,own_wr.avMagnitude,own_wr.frequency,turb.Ct_f,turb.Cp_f,K,turb.A))
    _,b1,c1 = result[0]
    return b1,c1,timings.best

def timed_analytical_aep():
    result = []
    timings = %timeit -o -q result.append(ntag_v02(r_jk,theta_jk,own_wr.cjd3_PA_all_coeffs,MEAN_CT, K,turb.A))
    _,b2,c2 = result[0]
    return b2,c2,timings.best

def timed_FYP_aep():
    result = []
    timings = %timeit -o -q result.append(ca_ag_v02(r_jk,theta_jk,own_wr.cjd_full_Fourier_coeffs_noCp,turb.Cp_f,MEAN_CT,K,turb.A,rho=1.225))
    _,b3,c3 = result[0] 
    return b3,c3,timings.best

def numerical_ff(): #Numerical "flow field"
    a4,_,_ = cubeAv_v4(r_jk,theta_jk,theta_i,own_wr.avMagnitude,own_wr.frequency,turb.Ct_f,turb.Cp_f,K,turb.A)
    return (0.5*turb.A*1.225*a4)/(1*10**6)

fi = FlorisInterface("floris_settings.yaml")

def nice_polar_plot(fig,gs,x,y,text):
    #first column is the wind rose
    ax = fig.add_subplot(gs,projection='polar')
    ax.plot(x,y,color='black',linewidth=2)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(60)  # Move radial labels away from plotted line
    ax.text(0, 0, text, ha='left',transform=ax.transAxes,color='black')
    ax.spines['polar'].set_visible(False)
    return None

# plot the data
def pce(exact,approx):
    return 100*(exact-approx)/exact

from AEP3_3_functions import si_fm

def nice_contourf_plot(site_label):
    ax1 = fig1.add_subplot(gs[2*i,1+j])
    ax1.set_aspect('equal')
    cf = ax1.contourf(X,Y,a4.reshape(X.shape),50,cmap=cm.coolwarm)
    ax1.scatter(xt,yt,marker='x',color='white')

    for k in range(Nt): #for each turbine
        #label that turbine with text of the power output and the percentage difference
        label_text = f'''N:{k}
        {b4[k]:.2f}MW
        {b1[k]:.2f}MW({pce(b1[k],b4[k]):+.2f}\%)
        {b2[k]:.2f}MW({pce(b2[k],b4[k]):+.2f}\%)
        {b3[k]:.2f}MW({pce(b3[k],b4[k]):+.2f}\%)'''
        ax1.text(xt[k],yt[k],label_text,fontsize=2,ha='center',va='center')

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    top_left_text = f'''Farm AEP Values (times slower)
    floris2:{c4:.2f}MW({pce(c4,c4):+.1f}\%) in {si_fm(d4)}s({d4/d1:.2f})
    florisM:{c0:.2f}MW({pce(c0,c4):+.1f}\%) in {si_fm(d0)}s({d0/d1:.2f})
    DsConv :{c1:.2f}MW({pce(c1,c4):+.1f}\%) in {si_fm(d1)}s({d1/d1:.2f})
    nctag  :{c2:.2f}MW({pce(c2,c4):+.1f}\%) in {si_fm(d2)}s({d2/d1:.2f})
    cubeAv :{c3:.2f}MW({pce(c3,c4):+.1f}\%) in {si_fm(d3)}s({d3/d1:.2f})'''
    ax1.text(0.5,1.0,top_left_text,color='black',transform=ax1.transAxes,va='top',ha='center',fontsize=4,bbox=props)

    ax1.text(0.02,0.02,site_label,color='black',transform=ax1.transAxes,va='bottom',ha='left',fontsize=4,bbox=props)
    
    cax = fig1.add_subplot(gs[2*i+1,1+j])
    cb = fig1.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
    
    return None

def nice_scatter_plot2():
    #performance chart
    ax2.scatter(Nt,d1,marker='o',s=20,color='black') #numerical
    ax2.scatter(Nt,d2,marker='x',s=20,color='red') 
    ax2.scatter(Nt,d3,marker='+',s=20,color='blue')
    return None

def nice_scatter_plot3(): 
    #accuracy chart
    ax3.scatter(Nt,np.abs(pce(c1,c0)),marker='o',s=10,color='black') #numerical
    ax3.scatter(Nt,np.abs(pce(c2,c0)),marker='x',s=10,color='red') 
    ax3.scatter(Nt,np.abs(pce(c3,c0)),marker='+',s=10,color='blue') 
    return None


#initalise the figure
cmtoI = 1/2.54  # centimeters to inches
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
gs = GridSpec(len(sites)*2, len(square_no)+1, height_ratios=[1,0.05]*len(sites),wspace=0.1,hspace=0.3)
import matplotlib.pyplot as plt
from matplotlib import cm
FIG_SIZE = 15
fig1 = plt.figure(figsize=(FIG_SIZE*len(square_no)*cmtoI,FIG_SIZE*len(sites)*cmtoI),dpi=250)
fig2 = plt.figure(figsize=(3*FIG_SIZE*cmtoI,3*FIG_SIZE*cmtoI),dpi=250)
ax2 = fig2.add_subplot()
fig3 = plt.figure(figsize=(3*FIG_SIZE*cmtoI,3*FIG_SIZE*cmtoI),dpi=250)
ax3 = fig3.add_subplot()

for i in range(len(sites)): #for each wind rose (row)    
    own_wr,MEAN_CT = get_own_wind_rose(sites[i]) #wind rose for my methods
    initalise_floris_wind_rose(sites[i])
    #first column is the wind rose
    nice_polar_plot(fig1,gs[2*i,0],theta_i,turb.Cp_f(own_wr.avMagnitude)*own_wr.avMagnitude*own_wr.frequency,'$C_p(U[\\theta])P[\\theta]U[\\theta]$')
    
    for j in range(len(square_no)): #for each layout
        if j< ti: #first ones are square layouts
            xt,yt,layout,Nt = rectangular_layout(square_no[j],square_no[j],s=SPACING)
            site_label = f"site:{sites[i]}, k:{K}, spacing: {SPACING}D,Nt={Nt}, turbine: {turb.__class__.__name__},terms:36"
        else: #the last ones columns are random layouts
            min_spacing = SPACING - 1
            xt,yt,layout,Nt,mean_dist = random_layout(square_no[j],spacing=min_spacing)
            site_label = f"site:{sites[i]}, k:{K}, min_spacing: {min_spacing}D,{square_no[j]}x{square_no[j]}(N={Nt},AvNearest_spacing={mean_dist:.2f}) turbine: {turb.__class__.__name__},terms:36"
        #get the domain
        X,Y,plot_points = rectangular_domain(layout,s=SPACING,pad=5,r=RESOLUTION)

        ### AEP calculation ###
        c0,d0 = timed_floris_aep()
        b4,c4,d4 = timed_floris_aep2()

        r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
        with np.errstate(all='ignore'): #turn off spam
            b1,c1,d1 = timed_numerical_aep()
            b2,c2,d2 = timed_analytical_aep()
            b3,c3,d3 = timed_FYP_aep()

        ### Flow field ###
        r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)

        a4 = numerical_ff()

        ### contour plot + illustration ###
        nice_contourf_plot(site_label)
        nice_scatter_plot2() #performance chart
        nice_scatter_plot3() #accuracy chart


        print(f"==== layout {j+1} of {len(square_no)} with site {i+1} of {len(sites)} complete")

ax3.set_xlabel('Nt')
ax3.set_ylabel('Accuracy')
ax2.set_xlabel('Nt')
ax2.set_ylabel('Runtime')

if SAVE_FIG:
    #% Save the figures
    from pathlib import Path
    path = "JFM_report_v01/Figures/"+Path(__file__).stem
    contour_name = path+"_"+str(SPACING)+"_"+"CONTOUR_"+"D_site.png"
    fig1.savefig(contour_name,dpi='figure',format='png',bbox_inches="tight")
    perf_scatter = path+"_"+str(SPACING)+"_"+"SCATTER1_"+"D_site.png"
    fig2.savefig(perf_scatter,dpi='figure',format='png',bbox_inches="tight")
    perf_scatter = path+"_"+str(SPACING)+"_"+"SCATTER2_"+"D_site.png"
    fig3.savefig(perf_scatter,dpi='figure',format='png',bbox_inches="tight")
