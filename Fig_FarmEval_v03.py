#%% Figure to show the effect of things on aep accuracy
#totally different from v01

#v02 was more 

#rows: effect of changes to the wind rose
#colums: effect of increasing the size of the farm

#the plotting is based on Fig_XtermAprx_v03, the data generation is based on Fig_thrstCoeffAprx_v07

#% A figure to show the effect of neglecting the cross terms in the product of the sums
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
                       Ct_op=3,Cp_op=3,cross_ts=True,ex=False)
    return pow_j,Uwff_ja

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

X,Y,plot_points = rectangular_domain(EXTENT,r=RESOLUTION)

Uwff_c = np.zeros((ROWS,COLS,plot_points.shape[0]))

powj_a,time_a,powj_b,time_b,powj_c,time_c,powj_d,layout= [empty2dPyarray(ROWS, COLS) for _ in range(8)]  #2d python arrays

U_i,P_i = [np.zeros((NO_BINS,len(site_n))) for _ in range(2)]

#generate the contourf data
from AEP3_3_functions import num_F_v02,simple_Fourier_coeffs_v01
for i in range(ROWS): #for each wind rose
    U_i[:,i],P_i[:,i] = get_floris_wind_rose(site_n[i])
    _,Fourier_coeffs_PA = simple_Fourier_coeffs_v01(turb.Cp_f(U_i[:,i])*(P_i[:,i]*(U_i[:,i]**3)*len(P_i[:,i]))/(2*np.pi))
    WAV_CT = np.sum(turb.Ct_f(U_i[:,i])*P_i[:,i])
    print("WAV_CT: {}".format(WAV_CT))
    for j in range(COLS): #for each layout
        
        layout[i][j] = rectangular_layout(layout_n[j],SPACING,rot[j])
        #floris aep (the reference)
        powj_a[i][j],time_a[i][j] = floris_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,timed=False)

        #analytical aep
        powj_b[i][j],time_b[i][j] = analytical_timed_aep(Fourier_coeffs_PA,layout[i][j],WAV_CT,K,turb,timed=False)
        
        #numerical aep (for flow field and as a reference)
        powj_c[i][j],Uwff_c[i,j,:] = numerical_aep(U_i[:,i],P_i[:,i],np.deg2rad(theta_i),layout[i][j],plot_points,turb,K)

        #floris NO WAKE aep
        powj_d[i][j],_ = floris_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,wake=False)
        
        print(f"{COLS*i+(j+1)}/{ROWS*COLS}\r")
        
#%% sanity check
n,m = 0,0
Nt =  layout[n][m].shape[0]
alpha = ((0.5*1.225*turb.A)/(1*10**6))
no_wake_p = Nt*alpha*np.sum(P_i[:,n]*turb.Cp_f(U_i[:,n])*U_i[:,n]**3)
print("no_wake_p: {}".format(no_wake_p))
#this is approximately correct ...
#%%plot the data ...
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

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
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)
    ax.annotate(ann_txt, xy=(0.4,0.75), ha='center', va='bottom',color='black',xycoords='axes fraction',rotation='vertical',bbox=props)
    ax.spines['polar'].set_visible(False)
    return None

from AEP3_3_functions import si_fm
from matplotlib import cm
def nice_composite_plot_v03(fig,gs,i,j,Z1,X,Y,Z2,xt,yt,cont_lim=(None,None)):
    ax = fig.add_subplot(gs[2*i,j+1])

    xticks = ax.xaxis.get_major_ticks()
    xticks[2].set_visible(False)
    ax.set_xlabel('$x/d_0$',labelpad=-9)

    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-19)
    #contourf
    vmin,vmax = cont_lim
    cf = ax.contourf(X,Y,Z1,50,cmap=cmap1,vmin=vmin,vmax=vmax)
    #scatter plot
    color_list = plt.cm.coolwarm(np.linspace(0, 1, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(color_list)
    sp = ax.scatter(xt,yt,c=Z2,cmap=cmap,marker='x',s=10)
    cax = fig.add_subplot(gs[2*i+1,j+1])
    cb = fig.colorbar(sp, cax=cax, orientation='horizontal',format='%.3g')
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set(xlim=(-EXTENT, EXTENT), ylim=(-EXTENT, EXTENT))
    # cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')

    #Then the farm total values:
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,pad=0.1)
    aep_a = np.sum(powj_a[i][j]) #floris reference
    aep_b = np.sum(powj_b[i][j]) #analytical AEP directly
    aep_c = np.sum(powj_c[i][j]) #AEP from cubed weight average velocity
    aep_d = np.sum(powj_d[i][j]) #no wake reference
    top_left_text = f'''{aep_a:.2f}MW(ref)
    {aep_b:.2f}MW({pce(aep_a,aep_b):+.1f}\%) in {si_fm(time_b[i][j])}s({time_a[i][j]/time_b[i][j]:.2f})
    {aep_c:.2f}MW({pce(aep_a,aep_c):+.1f}\%)
    {aep_d:.2f}MW({pce(aep_a,aep_d):+.1f}\%)'''

    ax.text(0.05,0.95,top_left_text,color='black',transform=ax.transAxes,va='top',ha='left',fontsize=4,bbox=props)

    if i == 4 and j == 2:
        cax.set_xlabel('Percentage Error in AEP / \%',labelpad=2)
        
    return cf

def ill_cb(gs,cont_lim):
    cax = fig.add_subplot(gs)
    #cax.invert_xaxis()

    import matplotlib as mpl
    cmap = cmap1
    vmin,vmax = cont_lim
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax, orientation='horizontal',label='$\\overline{U_w}$')
    
    return None

def pce(exact,approx):
    return 100*(approx-exact)/exact

def Cp_plot(gs,cont_lim): #the plot of the variation in power coefficient
    ax = fig.add_subplot(gs)
    ax.set(xlim=cont_lim)
    xs = np.linspace(cont_lim[0],cont_lim[1],200)
    ys = turb.Cp_f(xs)
    ax.plot9
    return None

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
gs = GridSpec(7, 4, height_ratios=[14,1,14,1,14,1,1],wspace=0.3,hspace=0.41)
fig = plt.figure(figsize=(7.8,8), dpi=300)

cont_lim = (np.min(Uwff_c),np.max(Uwff_c))

from matplotlib.colors import LinearSegmentedColormap
colors = ["black", "white"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

for i in range(ROWS): 
    #first column is the wind rose
    y1 = U_i[:,i]*P_i[:,i]
    nice_polar_plot(fig,gs[2*i,0],np.deg2rad(theta_i),y1,"$P(\\theta)U(\\theta)$")
    for j in range(COLS): #then onto the contours
        Z2 = pce(powj_a[i][j], powj_b[i][j])
        xt,yt = layout[i][j][:,0],layout[i][j][:,1]
        cf = nice_composite_plot_v03(fig,gs,i,j,Uwff_c[i][j].reshape(X.shape),X,Y,Z2,xt,yt,cont_lim=cont_lim) 

ill_cb(gs[6,1:],cont_lim) #'illustrative' colourbar on bottom row

plt.show()

if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')
    print("figure saved")
        
#%% arsing around with Taylor series expansion of Cp
U = U_i[:,2]
x1 = U**3
y1 = turb.Cp_f(U)

A = np.vstack([x1, np.ones(len(x1))]).T
m,c = np.linalg.lstsq(A, y1, rcond=None)[0]

import matplotlib.pyplot as plt
plt.plot(x1, y1, 'o', label='Original data', markersize=4)
plt.plot(x1, m*x1 + c, 'r', label='Fitted line')
plt.legend()
plt.show()

#%%
x2 = U_i[:,2]**3
plt.plot(x1, y1, 'o', label='Original data', markersize=10)
plt.plot(x1, m*x1 + c, 'r', label='Fitted line')
plt.legend()
plt.show()