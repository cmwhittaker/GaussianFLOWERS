#%% A figure to show the per-turbine AEP, wind rose, and wake velocity field. (To evaluate the model)
%load_ext autoreload
%autoreload 2

import numpy as np
no_xt = 1 #number of turbines in x
no_yt = 1
SITE = 7
CUSTOM = None #|None Nothing | 1 Uniform | 2 impulse
SPACING = 7
K = 0.025

SAVE_FIG = True

def rectangular_layout(s=5):
    xt = np.arange(1,no_xt+1,1)*s
    yt = np.arange(1,no_yt+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)
    return Xt.reshape(-1),Yt.reshape(-1),np.column_stack((Xt.reshape(-1),Yt.reshape(-1))), np.size(Xt)#just a single layout for now

def rectangular_domain(layout,s=5,pad=3,r=200):
    Xt,Yt = layout[:,0],layout[:,1]
    pad = 1.0
    xr,yr = r,r #resolution
    X,Y = np.meshgrid(np.linspace(np.min(Xt)-pad*s,np.max(Xt)+pad*s,xr),np.linspace(np.min(Yt)-pad*s,np.max(Yt)+pad*s,yr))
    return X,Y,np.column_stack((X.reshape(-1),Y.reshape(-1)))

def calc_floris_powers(layout,wr_speed,wr_freq,D,wake=True):
    # a lil function to make the above more readable
    no_bins = wr_speed.size
    theta_i = np.linspace(0,360,no_bins,endpoint=False)
    Nt = layout.shape[0] #more readable
    pow_ij = np.zeros((no_bins,Nt))
    fi.reinitialize(layout_x=D*layout[:,0],layout_y=D*layout[:,1])
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
    pow_j = np.sum(pow_ij,axis=0)
    aep = np.sum(pow_ij)
    return pow_j,aep

from turbines_v01 import iea_10MW,iea_15MW
from AEP3_2_functions import y_5MW
turb = iea_10MW()
no_bins = 72
from distributions_vC05 import wind_rose
wr = wind_rose(bin_no_bins=no_bins,custom=CUSTOM,a_0=8,site=SITE,Cp_f=turb.Cp_f)
MEAN_CT = turb.Ct_f(np.sum(wr.frequency*wr.avMagnitude)) #the fixed CT value
theta_i = np.linspace(0,2*np.pi,no_bins,endpoint=False)

xt,yt,layout, Nt = rectangular_layout(s=SPACING)
X,Y,plot_points = rectangular_domain(layout,s=SPACING,pad=3,r=50)

from AEP3_3_functions import gen_local_grid_v01C,cubeAv_v4,ntag_PA_v02,ca_ag_v02

from floris.tools import FlorisInterface
fi = FlorisInterface("floris_settings.yaml")

## AEP ##
r_jk,theta_jk = gen_local_grid_v01C(layout,layout)
b0,c0 = calc_floris_powers(layout, #Floris AEP
                       wr.avMagnitude,
                       wr.frequency,
                       turb.D)
_,b1,c1 = cubeAv_v4(r_jk,theta_jk, #Numerical AEP
                     theta_i,
                     wr.avMagnitude,
                     wr.frequency,
                     turb.Ct_f,
                     turb.Cp_f,
                     K,turb.A)
_,b2,c2 = ntag_PA_v02(r_jk,theta_jk, #Analytical AEP
                    wr.cjd3_PA_all_coeffs,
                    MEAN_CT, 
                    K,turb.A)

_,_,ref_P = ca_ag_v02(r_jk,theta_jk, #FYP method as a reference
                       wr.cjd_full_Fourier_coeffs_noCp,
                       turb.Cp_f,
                       MEAN_CT,
                       K,turb.A,rho=1.225)

## Flow field ##
r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
a3,_,_ = cubeAv_v4(r_jk,theta_jk, #Numerical "flow field"
                     theta_i,
                     wr.avMagnitude,
                     wr.frequency,
                     turb.Ct_f,
                     turb.Cp_f,
                     K,turb.A)
a4,_,_ = ntag_PA_v02(r_jk,theta_jk, #Analytical "flow field"
                    wr.cjd3_PA_all_coeffs,
                    MEAN_CT, 
                    K,turb.A)

#%Plot that data

def pce(approx,exact,rel=0):
    #rel is a modification to make the error more realistic
    return 100*(exact-approx)/(exact-rel)

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

cmtoI = 1/2.54  # centimeters to inches
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 4, height_ratios=[1,0.05,0.3],width_ratios=[0.25,0.25,0.25,0.25],wspace=0.1,hspace=0.3)
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure(figsize=(19*cmtoI,19*cmtoI),dpi=200)
#was 19cm, other was 25
#first is the contourf
ax1 = fig.add_subplot(gs[0,:])
ax1.set_aspect('equal')
cf = ax1.contourf(X,Y,a3.reshape(X.shape),50,cmap=cm.coolwarm)

for j in range(Nt): #for each turbine
    #label that turbine with text of the power output and the percentage difference
    label_text = f'''N:{j}
    {b0[j]:.2f}MW
    {b1[j]:.2f}MW({pce(b1[j],b0[j]):+.2f}\%)
    {b2[j]:.2f}MW({pce(b2[j],b0[j]):+.2f}\%)'''
    ax1.text(xt[j],yt[j],label_text,fontsize=4,ha='center',va='center')
    
ax1.scatter(xt,yt,marker='x',color='white')

props = dict(boxstyle='round', facecolor='white', alpha=0.8)
top_left_text = f'''Farm AEP Values
floris:{c0:.2f}MW
DsConv:{c1:.2f}MW({pce(c1,c0):+.2f}\%)
AnalyG:{c2:.2f}MW({pce(c2,c0):+.2f}\%)
Old(!):{ref_P:.2f}MW({pce(ref_P,c0):+.2f}\%)'''
ax1.text(0.1,0.92,top_left_text,color='black',transform=ax1.transAxes,va='center',ha='center',fontsize=6,bbox=props)

top_right_text = f'''Turbine label keys
Turbine number
Floris power
Numerical power
Analytical power'''

ax1.text(0.8,0.92,top_right_text,color='black',transform=ax1.transAxes,va='center',ha='center',fontsize=6,bbox=props)

ax1.text(0.02,0.02,"site:{}, k:{}, spacing: {}D, turbine: {}".format(SITE,K,SPACING,turb.__class__.__name__),color='black',transform=ax1.transAxes,va='bottom',ha='left',fontsize=10,bbox=props)

#then the colourbar
cax = fig.add_subplot(gs[1,:])
cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
#then the probability
xs = np.linspace(0,2*np.pi,no_bins,endpoint=False)
nice_polar_plot(fig,gs[2,0],xs,wr.frequency,'$P[\\theta]$')
#then the speed
nice_polar_plot(fig,gs[2,1],xs,wr.avMagnitude,'$U[\\theta]$')
#the joint
nice_polar_plot(fig,gs[2,2],xs,turb.Cp_f(wr.avMagnitude)*wr.avMagnitude*wr.frequency,'$C_p(U[\\theta])P[\\theta]U[\\theta]$')
#then Ct variation
nice_polar_plot(fig,gs[2,3],xs,turb.Ct_f(wr.avMagnitude),'$C_t(U[\\theta])$')

if SAVE_FIG:
    #% Save the figures
    from pathlib import Path
    path_plus_name = "JFM_report_v01/Figures/"+Path(__file__).stem+"_"+str(no_xt) + "x" + str(no_yt)+"_"+str(SPACING)+"D_site" +str(SITE) + ".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches="tight")

#%% set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)
rc('text.latex', preamble='\usepackage{color}')

#%% Tagging on to do some performance testing
%timeit a4,_,_ = ntag_PA_v02(r_jk,theta_jk,wr.cjd3_PA_all_coeffs,MEAN_CT, K,turb.A)
%timeit a4,_,_ = ntag_v02(r_jk,theta_jk,wr.cjd3_full_Fourier_coeffs,MEAN_CT, K,turb.A)
%timeit _,b1,c1 = cubeAv_v4(r_jk,theta_jk,theta_i,wr.avMagnitude,wr.frequency,turb.Ct_f,turb.Cp_f,K,turb.A)
#%% sanity check
from AEP3_3_functions import ntag_PA_v02
a41,_,_ = ntag_PA_v02(r_jk,theta_jk,wr.cjd3_PA_all_coeffs,MEAN_CT, K,turb.A)
a42,_,_ = ntag_v02(r_jk,theta_jk,wr.cjd3_full_Fourier_coeffs,MEAN_CT, K,turb.A)

#%% testing coordinate alginment (it's correct!)
from AEP3_3_functions import ca_ag_v02
U_ref,_,_ = ca_ag_v02(r_jk,theta_jk, 
                       wr.cjd_full_Fourier_coeffs_noCp,
                       turb.Cp_f,
                       MEAN_CT,
                       K,turb.A,rho=1.225)

import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
cf = ax.contourf(X,Y,U_ref.reshape(X.shape),50,cmap=cm.coolwarm)
fig.colorbar(cf)