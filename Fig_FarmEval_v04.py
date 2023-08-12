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

def slow_numerical_aep(U_i,P_i,theta_i,layout,plot_points,turb,K):
    pow_j,_,Uwff_ja= num_F_v02(U_i,P_i,theta_i,
                     layout,
                     plot_points,
                     turb,
                     K=K,
                     u_lim=None,
                     Ct_op=1,WAV_CT=WAV_CT,
                     Cp_op=1,cross_ts=True,ex=False)
    return pow_j,Uwff_ja

def fast_numerical_aep(U_i,P_i,theta_i,layout,turb,K,timed=True):
    from AEP3_3_functions import cubeAv_v5
    if timed:
        result = []
        timings = %timeit -o -q result.append(cubeAv_v5(U_i,P_i,theta_i,layout,layout,turb,RHO=1.225,K=K,u_lim=None,ex=False,Ct_op=2,WAV_CT=WAV_CT,Cp_op=1))
        pow_j,_ = result[0]
        time = timings.best
    else:
        pow_j = cubeAv_v5(U_i,P_i,theta_i,layout,layout,turb,RHO=1.225,K=K,u_lim=None,ex=False,Ct_op=2,WAV_CT=WAV_CT,Cp_op=1)
        time = np.NaN
    
    return pow_j,time

def ntag_timed_aep(Fourier_coeffs3_PA,layout,WAV_CT,K,turb,timed=True):
    from AEP3_3_functions import ntag_PA_v03
    if timed:
        result = []
        timings = %timeit -o -q result.append(ntag_PA_v03(Fourier_coeffs3_PA,layout,layout,turb,WAV_CT,K))
        pow_j,_ = result[0]
        time = timings.best
    else:
        pow_j,_ = ntag_PA_v03(Fourier_coeffs3_PA,layout,layout,turb,WAV_CT,K)
        time = np.NaN
    return pow_j,time

def caag_timed_aep(Fourier_coeffs_noCp_PA,layout,WAV_CT,K,turb,timed=True):
    from AEP3_3_functions import caag_PA_v03
    if timed:
        result = []
        timings = %timeit -o -q result.append(caag_PA_v03(Fourier_coeffs_noCp_PA,layout,layout,turb,WAV_CT,K))
        pow_j,_ = result[0]
        time = timings.best
    else:
        pow_j,_ = caag_PA_v03(Fourier_coeffs_noCp_PA,layout,layout,turb,WAV_CT,K)
        time = np.NaN
    return pow_j,time

K = 0.03
U_LIM = None
NO_BINS = 72 #number of bins in the wind rose
theta_i = np.linspace(0,360,NO_BINS,endpoint=False)

ROWS = 3
COLS = 3

from turbines_v01 import iea_10MW
turb = iea_10MW()

site_n = [2,3,6] #[2,3,6] #[6,8,10] are the tricky ones
layout_n = [5,6,7]
rot = [0,0,0]

X,Y,plot_points = rectangular_domain(EXTENT,r=RESOLUTION)

layout,powj_a,powj_b,powj_c,powj_d,powj_e,powj_f= [empty2dPyarray(ROWS, COLS) for _ in range(7)]  #2d python arrays
time_a,time_c,time_d,time_e = [np.zeros((ROWS,COLS)) for _ in range(4)]

Uwff_b = np.zeros((ROWS,COLS,plot_points.shape[0]))

U_i,P_i = [np.zeros((NO_BINS,len(site_n))) for _ in range(2)]

#generate the contourf data
from AEP3_3_functions import num_F_v02,simple_Fourier_coeffs_v01
for i in range(ROWS): #for each wind rose
    U_i[:,i],P_i[:,i] = get_floris_wind_rose(site_n[i])
    _,Fourier_coeffs3_PA = simple_Fourier_coeffs_v01(turb.Cp_f(U_i[:,i])*(P_i[:,i]*(U_i[:,i]**3)*len(P_i[:,i]))/(2*np.pi))
    _,Fourier_coeffs_noCp_PA = simple_Fourier_coeffs_v01((P_i[:,i]*U_i[:,i]*len(P_i[:,i]))/(2*np.pi))
    
    WAV_CT = np.sum(turb.Ct_f(U_i[:,i])*turb.Cp_f(U_i[:,i])*P_i[:,i]*U_i[:,i]**3/np.sum(turb.Cp_f(U_i[:,i])*P_i[:,i]*U_i[:,i]**3))

    for j in range(COLS): #for each layout
        timed = True
        layout[i][j] = rectangular_layout(layout_n[j],SPACING,rot[j])
        #floris aep (the reference)
        powj_a[i][j],time_a[i][j] = floris_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,timed=timed)

        #numerical aep (for flow field and as a reference)
        powj_b[i][j],Uwff_b[i,j,:] = slow_numerical_aep(U_i[:,i],P_i[:,i],np.deg2rad(theta_i),layout[i][j],plot_points,turb,K)

        #numerical aep Ct(average), Cp(global)
        powj_c[i][j],time_c[i][j] = fast_numerical_aep(U_i[:,i],P_i[:,i],np.deg2rad(theta_i),layout[i][j],turb,K)

        #ntag (average of the cube) analytical aep
        powj_d[i][j],time_d[i][j] = ntag_timed_aep(Fourier_coeffs3_PA,layout[i][j],WAV_CT,K,turb,timed=timed)

        #caag (cube of the average) analytical aep
        powj_e[i][j],time_e[i][j] = caag_timed_aep(Fourier_coeffs_noCp_PA,layout[i][j],WAV_CT,K,turb,timed=timed)
        
        #floris NO WAKE aep
        powj_f[i][j],_ = floris_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,wake=False)      
        
        print(f"{COLS*i+(j+1)}/{ROWS*COLS}\r")
#%%
#ntag (average of the cube) analytical aep
d = Fourier_coeffs3_PA
a,b = ntag_timed_aep(d,c,WAV_CT,K,turb,timed=True)
print(np.sum(a))
print(f"{si_fm(b)}s")

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
    #sp = ax.scatter(xt,yt,marker='x',s=10,c='black')
    cax = fig.add_subplot(gs[2*i+1,j+1])
    cb = fig.colorbar(sp, cax=cax, cmap=cmap,orientation='horizontal',format='%.3g')
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set(xlim=(-EXTENT, EXTENT), ylim=(-EXTENT, EXTENT))

    #Then the farm total values:
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,pad=0.1)

    aep_a = np.sum(powj_a[i][j]) #floris reference
    aep_b = np.sum(powj_b[i][j]) #analytical AEP directly
    aep_c = np.sum(powj_c[i][j]) #AEP from cubed weight average velocity
    aep_d = np.sum(powj_d[i][j]) #no wake reference
    aep_e = np.sum(powj_e[i][j]) #no wake reference
    aep_f = np.sum(powj_f[i][j]) #no wake reference

    top_left_text = f'''{aep_a:.2f}MW(ref) in {si_fm(time_a[i][j])}s
    {aep_b:.2f}MW({pce(aep_a,aep_b):+.1f}\%) 
    {aep_c:.2f}MW({pce(aep_a,aep_c):+.1f}\%) in {si_fm(time_c[i][j])}s({time_a[i][j]/time_c[i][j]:.2f})
    {aep_d:.2f}MW({pce(aep_a,aep_d):+.1f}\%) in {si_fm(time_d[i][j])}s({time_a[i][j]/time_d[i][j]:.2f})
    {aep_e:.2f}MW({pce(aep_a,aep_e):+.1f}\%) in {si_fm(time_e[i][j])}s({time_a[i][j]/time_e[i][j]:.2f})
    {aep_f:.2f}MW({pce(aep_a,aep_f):+.1f}\%)'''

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

from AEP3_3_functions import pce

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

cont_lim = (np.min(Uwff_b),np.max(Uwff_b))

from matplotlib.colors import LinearSegmentedColormap
colors = ["black", "white"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

for i in range(ROWS): 
    #first column is the wind rose
    y1 = U_i[:,i]*P_i[:,i]
    nice_polar_plot(fig,gs[2*i,0],np.deg2rad(theta_i),y1,"$P(\\theta)U(\\theta)$")
    for j in range(COLS): #then onto the contours
        Z2 = pce(powj_a[i][j], powj_d[i][j])
        xt,yt = layout[i][j][:,0],layout[i][j][:,1]
        cf = nice_composite_plot_v03(fig,gs,i,j,Uwff_b[i][j].reshape(X.shape),X,Y,Z2,xt,yt,cont_lim=cont_lim) 

ill_cb(gs[6,1:],cont_lim) #'illustrative' colourbar on bottom row

if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')
    print("figure saved")

plt.show()
        
#%% trying to sanity check
from AEP3_3_functions import cubeAv_v5,num_F_v02,ntag_PA_v03,ntag_PA_TS_v01
#this should be the same as:
turb = iea_10MW()
aep6,_,Uff = num_F_v02(U_i1,P_i1,np.deg2rad(theta1_i),
                     layout1,
                     layout1,
                     turb,
                     K=K,
                     u_lim=None,
                     Ct_op=1,WAV_CT=WAV_CT,
                     Cp_op=1,cross_ts=True,ex=True)
print("reference: {}".format(np.sum(aep6)))
WAV_CT = np.sum(turb.Ct_f(U_i1)*turb.Cp_f(U_i1)*P_i1*U_i1**3/np.sum(turb.Cp_f(U_i1)*P_i1*U_i1**3))
WAV_CP = np.sum(turb.Cp_f(U_i1)*turb.Cp_f(U_i1)*P_i1*U_i1**3/np.sum(turb.Cp_f(U_i1)*P_i1*U_i1**3))
# WAV_CT = np.sum(turb.Ct_f(U_i1)*P_i1)

aep7,_,Uff = num_F_v02(U_i1,P_i1,np.deg2rad(theta1_i),
                     layout1,
                     layout1,
                     turb,
                     K=K,
                     u_lim=None,
                     Ct_op=3,WAV_CT=WAV_CT,
                     Cp_op=3,WAV_CP=WAV_CP,cross_ts=False,ex=False)
print("est ntag: {}".format(np.sum(aep7)))#
from AEP3_3_functions import ntag_PA_v03
_,fc_PA_d = simple_Fourier_coeffs_v01((turb.Cp_f(U_i1)*P_i1*(U_i1**3)*len(P_i1))/(2*np.pi))
aep8,_ = ntag_PA_v03(fc_PA_d,layout1,layout1,turb,WAV_CT,K)
print("actual ntag: {}".format(np.sum(aep8)))
turb = iea_10MW()
_,fc_PA_a = simple_Fourier_coeffs_v01((P_i1*(U_i1**4)*len(P_i1))/(2*np.pi))
_,fc_PA_b = simple_Fourier_coeffs_v01((P_i1*(U_i1**3)*len(P_i1))/(2*np.pi))
m,c = m_opt,c_opt
aep12,_ = ntag_PA_TS_v01(fc_PA_a,fc_PA_b,m,c,layout1,layout1,turb,WAV_CT,K)
print("anly new: {}".format(np.sum(aep12)))

#%% arsing around with Taylor series expansion
i = 0
layout1 = layout[0][2]
def new_wr1(NO_BINS):
        if not NO_BINS%4 == 0:
              raise ValueError("Must be neatly divisible by 4") 
        theta_i = np.deg2rad(np.linspace(0,360,NO_BINS,endpoint=False))
        U_inf = 13
        U_i = U_inf*np.ones(NO_BINS)
        P_i = np.zeros(NO_BINS)
        P_i[0], P_i[NO_BINS//4] = 0.5, 0.5
        return theta_i,U_i, P_i


U_i1, P_i1 = U_i[:,i],P_i[:,i]
theta1_i = np.linspace(0,360,NO_BINS,endpoint=False)
#theta_i, U_i1, P_i1 = new_wr1(360)


import numpy as np
import matplotlib.pyplot as plt

x1 = U_i1
from turbines_v01 import iea_10MW
turb = iea_10MW()
y1 = turb.Cp_f(U_i1)

#weights = P_i1 #weight by power generation
weights = P_i1 * U_i1**3
# Perform linear regression
m, c = np.polyfit(x1, y1, deg=1,w=weights)

# Generate the fitted line
fit_line = m * x1 + c
print(f"m: {m:.2f},c: {c:.2f}")

yf = lambda x: m*x+c

y6 = P_i1*turb.Cp_f(U_i1)*U_i1**3
print("np.sum(y6): {}".format(np.sum(y6)))
y7 = P_i1*yf(U_i1)*U_i1**3
print("np.sum(y7): {}".format(np.sum(y7)))

from scipy.optimize import minimize,differential_evolution
# define your objective function
def objective(p, U_i1, P_i1):
    m, c = p
    diff = P_i1*turb.Cp_f(U_i1)*U_i1**3 - P_i1*(m*U_i1 + c)*U_i1**3
    return np.sum(np.abs(diff))   # the sum of squared differences

# initial guess
p0 = [-0.04, 1]

from scipy.optimize import dual_annealing

bounds = list(zip([-1, -1], [1, 1])) # provide bounds for m and c
res = differential_evolution(objective, bounds, args=(U_i1, P_i1))

# optimal parameters
m_opt, c_opt = res.x

yf = lambda x: m_opt*x + c_opt

print("Optimal parameters are m={}, c={}".format(m_opt, c_opt))

y6 = P_i1*turb.Cp_f(U_i1)*U_i1**3
print("np.sum(y6): {}".format(np.sum(y6)))
y7 = P_i1*yf(U_i1)*U_i1**3
print("np.sum(y7): {}".format(np.sum(y7)))

#%%
# Plot the original data points and the fitted line
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.scatter(x1, y1, label='Data Points')
ax.set(xlim=[0,None])
ax2 = ax.twinx()
ax.plot(x1, fit_line, color='r', label='Fitted Line')
# Add legend to the plot
ax.legend()
             
aep11,_ = ntag_PA_v03(Fourier_coeffs3_PA,layout1,layout1,turb,WAV_CT,K)

print("aep11: {}".format(np.sum(aep11)))
#%%
pow1 = (P_i1*turb.Cp_f(U_i1)*U_i1**3)
pow2 = (P_i1*(m*U_i1+c)*U_i1**3)
print("pow1: {}".format(np.sum(pow1)))
print("pow2: {}".format(np.sum(pow2)))

#%%
#%%
turb = iea_10MW()
aep6,_,Uff = num_F_v02(U_i1,P_i1,np.deg2rad(theta1_i),
                     layout1,
                     layout1,
                     turb,
                     K=K,
                     u_lim=None,
                     Ct_op=1,WAV_CT=WAV_CT,
                     Cp_op=1,cross_ts=True,ex=False)
print("aep6: {}".format(np.sum(aep6)))

aep10,_ = cubeAv_v5(U_i1,P_i1,theta1_i,
              layout1,
              layout1, 
              turb,
              RHO=1.225,K=K,
              u_lim=None,ex=True,Ct_op=3,WAV_CT=WAV_CT)

print("aep10: {}".format(np.sum(aep10)))
_,fc_PA_1 = simple_Fourier_coeffs_v01(turb.Cp_f(U_i1)*(P_i1*(U_i1**3)*len(P_i1))/(2*np.pi))
aep11,_ = ntag_PA_v03(fc_PA_1,layout1,layout1,turb,WAV_CT,K)
print("aep11: {}".format(np.sum(aep11)))
_,fc_PA_2 = simple_Fourier_coeffs_v01((P_i1*(U_i1**3)*len(P_i1))/(2*np.pi))
# m1 = 0
WAV_CP = np.sum(P_i1*turb.Cp_f(U_i1))
m1 = m
c1 = c
aep12,_ = ntag_PA_TS_v01(fc_PA_2,m1,c1,layout1,layout1,turb,WAV_CT,K)
print("new aep12: {}".format(np.sum(aep12)))
_,fc_PA_3 = simple_Fourier_coeffs_v01(WAV_CP*(P_i1*(U_i1**3)*len(P_i1))/(2*np.pi))
aep13,_ = ntag_PA_v03(fc_PA_3,layout1,layout1,turb,WAV_CT,K)
print("ntag WAV_CP aep13: {}".format(np.sum(aep13)))
_,fc_PA_5 = simple_Fourier_coeffs_v01(turb.Cp_f(U_i1)*(P_i1*(U_i1**3)*len(P_i1))/(2*np.pi))
aep15,_ = ntag_PA_v03(fc_PA_5,layout1,layout1,turb,WAV_CT,K)
print("ntag aep15: {}".format(np.sum(aep15)))
from AEP3_3_functions import caag_PA_v03
_,fc_PA_4 = simple_Fourier_coeffs_v01((P_i1*(U_i1)*len(P_i1))/(2*np.pi))
aep14,_ = caag_PA_v03(fc_PA_4,layout1,layout1,turb,WAV_CT,K,Cp_op=2,WAV_CP=WAV_CP)
print("FYP WAVED aep14: {}".format(np.sum(aep14)))
aep15,_ = caag_PA_v03(fc_PA_4,layout1,layout1,turb,WAV_CT,K,Cp_op=1)
print("FYP aep15: {}".format(np.sum(aep15)))
#%%
import matplotlib.pyplot as plt
from matplotlib import cm
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
cf = ax.contourf(X,Y,Uff.reshape(X.shape),50,cmap=cm.coolwarm)
fig.colorbar(cf)