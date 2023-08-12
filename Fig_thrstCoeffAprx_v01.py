#% A figure to show the effect of setting the thrust coefficient as a constant based on the mean weight-averaged velocity.
#(A 3 x 3 pannel of the wake velocity deficit error for a WIND ROSE input)

#%% Set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

#%% get data to plot
import numpy as np

def rectangular_domain(extent,r=200):
    X,Y = np.meshgrid(np.linspace(-extent,extent,r),np.linspace(-extent,extent,r))
    return X,Y,np.column_stack((X.reshape(-1),Y.reshape(-1)))

K = 0.03
site_list=[6,3,2]
no_bins = 72
wr_list = []
fixed_Ct_list = []
Z_list = []
from distributions_vC05 import wind_rose
from AEP3_2_functions import y_5MW
turb = y_5MW()
from AEP3_3_functions import cubeAv_v4,gen_local_grid_v01C
#get domain
layout = np.array(((0,0),))
X,Y,plot_points = rectangular_domain(10,r=200)

for i in range(len(site_list)):
    #first and second columns are wind rose related
    wr = wind_rose(bin_no_bins=no_bins,custom=None,site=site_list[i],filepath=None,a_0=1,Cp_f=turb.Cp_f)
    wr_list.append(wr)
    #third column is the contour function
    r_jk,theta_jk = gen_local_grid_v01C(layout,plot_points)
    fixed_Cp_f = lambda x: 1 #power cofficient is fixed for now ... ? maybe revise later?
    actual_Z,_,_  = cubeAv_v4(r_jk,theta_jk,
                       np.linspace(0,2*np.pi,wr.bin_no_bins,endpoint=False),
                       wr.avMagnitude,
                       wr.frequency,
                       turb.Ct_f, #Thrust coefficient function
                       fixed_Cp_f,
                       K,
                       turb.A)
    fixed_Ct = turb.Ct_f(np.mean(wr.avMagnitude*wr.frequency*no_bins))
    fixed_Ct_list.append(fixed_Ct)
    fixed_Ct_f = lambda x: fixed_Ct
    approx_Z,_,_ = cubeAv_v4(r_jk,theta_jk,
                    np.linspace(0,2*np.pi,wr.bin_no_bins,endpoint=False),
                    wr.avMagnitude,
                    wr.frequency,
                    fixed_Ct_f, #fixed Ct
                    fixed_Cp_f,
                    K,
                    turb.A)
    pce = 100*(np.abs(actual_Z-approx_Z)/actual_Z)
    Z_list.append(pce)   
    
#%%plot the data ...
def nice_polar_plot(fig,gs,x,y,text):
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

def nice_cartesian_plot(fig,gs,x,y,ybar):
    ax = fig.add_subplot(gs)
    ax.plot(x,y,color='grey',linewidth=1)
    ax.hlines(y=ybar,xmin=0,xmax=360,linewidth=1,color='black',ls='--')

    annotation_txt = "$C_t(\overline{U_w})" + f'{ybar:.2f}$'    
    ax.annotate(annotation_txt, xy=(0, ybar), xytext=(0, 0),
             textcoords='offset points', ha='left', va='bottom')
    
    ax.set_xlabel('$\\theta$',labelpad=0)
    ax.set_ylabel('$C_t(\\theta)$',labelpad=0) 
    return None

from matplotlib import cm
def nice_contour_plot_v02(fig,gs,row,X,Y,Z,xt,yt):
    ax = fig.add_subplot(gs[row,2])
    cf = ax.contourf(X,Y,Z.reshape(X.shape),cmap=cm.coolwarm)
    #colourbar and decorations
    cax = fig.add_subplot(gs[row+1,2])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
    cb.ax.locator_params(nbins=6)
    cb.ax.tick_params(labelsize=9)
    ax.scatter(xt,yt,marker='x',color='black',s=10)

    return None

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(6, 3, height_ratios=[14,1,14,1,14,1],wspace=0.3,hspace=0.37)
fig = plt.figure(figsize=(7,7), dpi=200)

import numpy as np

for i in range(len(site_list)): #for each row
    #first column is the wind rose
    wr = wr_list[i]
    wind_rose_y = wr.frequency*wr.avMagnitude
    nice_polar_plot(fig,gs[2*i,0],np.deg2rad(wr.deg_bin_centers),wind_rose_y,text="$P(\\theta)U(\\theta)$")
    #second column is the thrust coefficient curve

    cnst_thrust_coeff = fixed_Ct_list[i]
    nice_cartesian_plot(fig,gs[2*i,1],wr.deg_bin_centers,turb.Ct_f(wr.avMagnitude),cnst_thrust_coeff)
    #third column is the percentage error contour plot
    nice_contour_plot_v02(fig,gs,2*i,X,Y,Z_list[i],0,0)

from pathlib import Path
path_plus_name = "JFM_report_v01/Figures/"+Path(__file__).stem+".png"
plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches="tight")

#%%

ks=np.array(([0.02,0.04,0.06]))
Cts=np.array(([0.7,0.8,0.9]))
eps = 0.2*np.sqrt((1+np.sqrt(1-Cts))/(2*np.sqrt(1-Cts))) 
x_lims = (np.sqrt(Cts/8)-eps)/ks

list_of_U_avs = []

list_of_list_of_DU_w_exact = [] #lol
list_of_list_of_errs = []
list_of_list_of_DU_w_approx = []
list_of_list_of_pc_error = []

list_of_cjd_func =[]

for j in range(3): #for each site

    bin_width = 5
    filepath = 'WindRoseData\\' +'site' +str(site_list[j])+'.csv'

    a = np.loadtxt(open(filepath,"rb"), delimiter=",", skiprows=2)

    bin_centres,frequency,avmagnitude,magMean = binWindData(a[:,5],a[:,6],bin_width)

    fourier_coeffs,cjd_func,U_av = cjd_from_binned_data(frequency,avmagnitude)

    list_of_U_avs.append(U_av)  

    list_of_DU_w_exact = []
    list_of_errs = []
    list_of_DU_w_approx = []
    list_of_pc_error = []

    for i in range(len(site_list)): #for each range of values
        points = 10
        x = np.array([np.linspace(-10,10,points)])
        y = np.array([np.linspace(-10,10,points)])
        #200x200 takes about 30min per turbine!
        X,Y = np.meshgrid(x,y)
        plot_coords = np.stack((X.reshape(-1),Y.reshape(-1)),axis=1)
        print(plot_coords.shape)
        params = ks[i],eps[i],Cts[i]

        DU_w_exact,err = ei_nt(plot_coords,np.array(([[0,0]])),params,cjd_func,feedback=True)
        DU_w_exact,err = DU_w_exact.reshape(X.shape), err.reshape(X.shape)
        DU_w_approx= ac_nt(plot_coords,np.array(([[0,0]])),params,fourier_coeffs).reshape(X.shape)

        pc_error = 100*np.abs((DU_w_exact-DU_w_approx)/DU_w_exact)

        list_of_DU_w_exact.append(DU_w_exact)
        list_of_errs.append(err)
        list_of_DU_w_approx.append(DU_w_approx)
        list_of_pc_error.append(pc_error)
    
    list_of_list_of_DU_w_exact.append(list_of_DU_w_exact) #lol
    list_of_list_of_errs.append(list_of_errs)
    list_of_list_of_DU_w_approx.append(list_of_DU_w_approx)
    list_of_list_of_pc_error.append(list_of_pc_error)

#%%Need to fix the pc_error
list_of_list_of_c_pc_error = []

for j in range(len(site_list)): #for each wind rose
    list_of_c_pc_error = []
    for i in range(len(ks)): #for each choice of model parameters
        
        U_w_exact = list_of_U_avs[j] - list_of_list_of_DU_w_exact[j][i]
        U_w_approx = list_of_U_avs[j] - list_of_list_of_DU_w_approx[j][i]
        #the "correct" percentage error
        c_pc_error = 100*np.abs((U_w_exact-U_w_approx)/U_w_exact)
        
        list_of_c_pc_error.append(c_pc_error)

    list_of_list_of_c_pc_error.append(list_of_c_pc_error)

#%%
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
thing = list_of_U_avs[2] - list_of_list_of_DU_w_exact[2][1]
print(list_of_U_avs[2])
cf = ax.contourf(X,Y,thing,50,cmap=cm.coolwarm)
fig.colorbar(cf)

#%%Then plot the results
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
gs = GridSpec(6, 4, height_ratios=[14,1,14,1,14,1],wspace=0.3,hspace=0.37)

fig = plt.figure(figsize=(7.8,7), dpi=300)

r_mins = [3,3,3]

R = np.sqrt(X**2+Y**2)

for j in range(3): #for each row
    
    list_of_pc_error =  list_of_list_of_c_pc_error[j]
    cjd_func = list_of_cjd_func[j]

    #first column is the wind rose
    ax = fig.add_subplot(gs[2*j,0],projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    xt=np.linspace(0,2*np.pi,1000)
    ax.plot(xt,cjd_func(xt),color='black')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(60)  # Move radial labels away from plotted line

    #the next 3 are the pc error contour plots
    for i in range(3):
        pc_error = list_of_pc_error[i]
        pc_error = np.where(R>3,pc_error,np.NaN)
        ax = fig.add_subplot(gs[2*j,i+1])
        cf = ax.contourf(X,Y,pc_error,10,cmap=cm.coolwarm)
        ax.set_xlabel('$x/d_0$',labelpad=-9)
        xticks = ax.xaxis.get_major_ticks()
        xticks[1].set_visible(False)

        text = '$k=' + str(ks[i]) + '$' + ',$C_T=' + str(Cts[i]) + '$'

        ax.text(0.98*np.max(x), 0.98*np.max(y), text, size=8, ha="right", va="top",bbox=dict(boxstyle="square",ec='w', fc='w',pad=0))

        ax.tick_params(axis='y', which='major', pad=1)
        if i==0:
            yticks = ax.yaxis.get_major_ticks()
            yticks[2].set_visible(False)
            ax.set_ylabel('$y/d_0$',labelpad=-19)
        cax = fig.add_subplot(gs[2*j+1,i+1])
        cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
        cb.ax.locator_params(nbins=5)
        cb.ax.tick_params(labelsize=8) 
        if i==1 and j==2:
            cax.set_xlabel('Absolute Percentage Error in $\overline{U_w}$ / \%',labelpad=5)

bbox = fig.bbox_inches.from_bounds(0.9,0.45,6.3,5.8) #Crop
plt.savefig(r"C:\Users\Work\OneDrive - Durham University\ENGI4093 Final Year Project\Python and Writing\Final Report\Figures\small_angle_error_V03.png",dpi='figure',format='png',bbox_inches=bbox)

#%% just looking at discontinuities in the wind roses

bin_width = 5
filepath = 'WindRoseData\\' +'site' +str(4)+'.csv'

a = np.loadtxt(open(filepath,"rb"), delimiter=",", skiprows=2)

bin_centres,frequency,avmagnitude,magMean = binWindData(a[:,5],a[:,6],bin_width)
fourier_coeffs,cjd_func,U_av = cjd_from_binned_data(frequency,avmagnitude)


import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=400)

xt = np.linspace(0,2*np.pi,1000)
ax.plot(xt,cjd_func(xt))

