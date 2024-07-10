#%% Produce the 3x3 wind roses(sites)xlayouts figure that compares the accuracy of Gaussian FLOWERS to a floris reference

# V2 simplifies the figure to only compare num int and GF

# there are 6 methods being evaluated at once (most will be removed before publishing)
# this "with errors" version adds an additional panel in the bottom left that shows
# the contribution of each of the assumptions on the final accuracy (assuming that they are independent - which is semi-true)

#(rows: effect of changes to the wind rose
#colums: effect of increasing the size of the farm)

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

U_LIM = 3 #manually override ("user limit") the invalid radius around the turbine (otherwise variable, depending on k/Ct) - 
RESOLUTION = 100 #number of x/y points in contourf meshgrid
EXTENT = 35 #total size of contourf "window" (square from -EXTENT,-EXTENT to EXTENT,EXTENT)
K = 0.03 #expansion parameter for the Gaussian model
Kj = 0.05 #expansion parameter for the Jensen model
NO_BINS = 72 #number of bins in the wind rose
ALIGN_WEST = False

SYNTHETIC_WR = True #(they are all synthetic)

FIG_VAR = 3 #Figure variation
from utilities.helpers import random_layouts
np.random.seed(1)
if FIG_VAR==1: #varying inflow velocities
    U_AVs = [6,10,12]
    site_var = [5,5,5]
    layouts = [random_layouts(1)[0],]*3
elif FIG_VAR==2: #varying density
    U_AVs = [10,10,10]
    site_var = [5,5,5]
    layouts = []
    widths = [54,42,30] #[70,52,38.5]
    min_rs = [6.455,5.084,3.73] #[8,6.3,4.6]
    for i in range(3):
        layouts.append(random_layouts(1,width=widths[i],min_r=min_rs[i])[0])
elif FIG_VAR==3: #varying size
    U_AVs = [10,10,10]
    site_var = [5,5,5]
    layouts = []
    widths = [32,42,54]
    min_rs = [5.08,5.08,5.08]
    for i in range(3):
        layouts.append(random_layouts(1,width=widths[i],min_r=min_rs[i])[0])
else:
    raise ValueError("Incorrect FIG_NO selected")

# #"meta" information about the layout used
# from sklearn.metrics.pairwise import euclidean_distances

# for i in range(3):
#     layout = layouts[i]
#     distances = euclidean_distances(layout,layout)
#     distances[distances<0.1]=np.nan #remove distance from point to itsel
#     m_nearest_dists= np.mean(np.nanmin(distances,axis=1))
#     print(f"N_turbs ={len(layout)},m_nearest_dists:{m_nearest_dists:.2f}")
    
ROWS = len(site_var) #number of sites
COLS = 1 #number of layout variations

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
    #get wind rose (NOT sorted using wind bearing)
    if SYNTHETIC_WR:
        U_i[:,i],P_i[:,i],_ = vonMises_wr(U_AVs[i],site_var[i])
    else:
        U_i[:,i],P_i[:,i],_,fl_wr = get_floris_wind_rose(site_var[i],align_west=ALIGN_WEST)

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
        
        # layout[i][j] = rectangular_layout(turb_n[i],spacing[i],np.deg2rad(rot[i]))
        layout[i][j] = layouts[i]
        
        #find the errors due to each assumption (numerically)
        errors[i,j,:] = find_errors(U_i[:,i],P_i[:,i],np.deg2rad(thetaD_i),plot_points,layout[i][j],turb,K)

        #floris aep (the reference)
        powj_a[i][j],time_a[i][j] = floris_AV_timed_aep(U_i[:,i],P_i[:,i],thetaD_WB_i,layout[i][j],turb,timed=timed)

        #non-vectorised numerical aep (flow field+aep)
        aep_func_b = lambda: num_Fs(U_i[:,i],P_i[:,i],np.deg2rad(thetaD_i),
                                    plot_points,layout[i][j],
                                    turb,K,
                                    u_lim=None,
                                    Ct_op=1, #local Ct
                                    Cp_op=1, #local Cp
                                    cross_ts=True,ex=True,
                                    ff=False)
        (powj_b[i][j],_,_),time_b[i][j] = adaptive_timeit(aep_func_b,timed=timed)

        #flow field (for visualisation - no timing) using num_F
        _,_,Uwff[i,j,:] = num_Fs(U_i[:,i],P_i[:,i],np.deg2rad(thetaD_i),
                                    plot_points,layout[i][j],
                                    turb,K,
                                    u_lim=None,
                                    Ct_op=1, #local Ct
                                    Cp_op=1, #local Cp
                                    cross_ts=True,ex=True,
                                    ff=True)


        #vectorised numerical aep (aep+time)
        aep_func_c = lambda: vect_num_F(U_i[:,i],P_i[:,i],
                                       np.deg2rad(thetaD_i),
                                       layout[i][j],layout[i][j],
                                       turb,
                                       K,
                                       u_lim=U_LIM,
                                       Ct_op=2, #global Ct
                                       Cp_op=1, #local Cp
                                       ex=True)
        (powj_c[i][j],_),time_c[i][j] = adaptive_timeit(aep_func_c,timed=timed)

        #ntag (No cross Terms Analytical Gaussian) (aep+time)
        aep_func_d = lambda: ntag_PA(Fourier_coeffs3_PA,
                                         layout[i][j],
                                         layout[i][j],
                                         turb,
                                         K, 
                                         #(Ct_op = 3 cnst) 
                                         #(Cp_op = 2 global )    
                                         wav_Ct)
        (powj_d[i][j],_),time_d[i][j] = adaptive_timeit(aep_func_d,timed=timed)

        #caag (cube of the average) analytical aep
        aep_func_e = lambda: caag_PA(Fourier_coeffs_noCp_PA,
                                         layout[i][j],
                                         layout[i][j],
                                         turb,
                                         K,
                                         #(Ct_op = 3 cnst) 
                                         #(Cp_op = 2 *local-ish)
                                         wav_Ct)
        # *local based on the weight averaged wake velocity 
        (powj_e[i][j],_),time_e[i][j] = adaptive_timeit(aep_func_e,timed=timed)

        # #floris NO WAKE aep (sanity check)
        powj_f[i][j],_ = floris_AV_timed_aep(U_i[:,i],P_i[:,i],thetaD_WB_i,layout[i][j],turb,wake=False,timed=False)   

        #flowers AEP
        aep_func_g = lambda: jflowers(Fourier_coeffs_j,
                                      layout[i][j],layout[i][j],
                                      turb,
                                      Kj,
                                      c_0,
                                      RHO=1.225,
                                      r_lim=0.5)
        (powj_g[i][j],_),time_g[i][j] = adaptive_timeit(aep_func_g,timed=timed)
        
        #floris AEP WITHOUT wind speed averaging
        # powj_h[i][j],time_h[i][j] = floris_FULL_timed_aep(fl_wr,layout[i][j],turb,timed=False)

        print(f"{COLS*i+(j+1)}/{ROWS*COLS}\r")

#%
#% Another ... version of the figure
#you need to run the cell above first

from utilities.plotting_funcs import si_fm
from utilities.helpers import pce
from matplotlib import cm

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

from matplotlib.colors import LinearSegmentedColormap
colors = ["black", "white"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

def nice_polar_plot(fig,gs,x,y,ann_txt,bar=True,wr_label=None):
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
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.2)
    ax.annotate(ann_txt, xy=(0.4,0.75), ha='center', va='bottom',color='black',xycoords='axes fraction',rotation='vertical',bbox=props)
    ax.spines['polar'].set_visible(False)
    if wr_label is not None:
        ax.annotate(wr_label, xy=(0,-0.05), ha='left', va='bottom',color='black',xycoords='axes fraction',rotation='horizontal',bbox=props,fontsize=9)
    return None

def nice_composite_plot_v03B(fig,cf_gs,ff_cb_gs,e_cb_gs,Z1,X,Y,Z2,xt,yt,errors,cont_lim=(None,None),ff_cb_label=False,e_cb_label=False,farm_label=None):
    ax = fig.add_subplot(cf_gs)

    xticks = ax.xaxis.get_major_ticks()
    xticks[2].set_visible(False)
    ax.set_xlabel('$x/D$',labelpad=-9)

    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/D$',labelpad=-19)
    ax.yaxis.set_tick_params(pad=0)
    
    #contourf
    vmin,vmax = cont_lim
    cf = ax.contourf(X,Y,Z1,50,cmap=cm.gray)

    #greyscale velocity colourbar
    ff_cb_ax = fig.add_subplot(ff_cb_gs)
    ff_cb = fig.colorbar(cf, cax=ff_cb_ax, cmap=cm.gray,orientation='horizontal',format='%.3g')
    from matplotlib.ticker import MaxNLocator
    ff_cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    if ff_cb_label:
        ff_cb.set_label("Average wind velocity / $ms^{-1}$")
    ax.set(xlim=(-EXTENT, EXTENT), ylim=(-EXTENT, EXTENT))

    #coloured scatter plot at turbine locations
    sp = ax.scatter(xt,yt,c=Z2,cmap=cm.viridis,marker='o',s=3,lw=1)
    e_cb_ax = fig.add_subplot(e_cb_gs)

    #scatter plot colourbar
    e_cb = fig.colorbar(sp, cax=e_cb_ax, cmap=cm.viridis,orientation='horizontal',format='%.3g') #the per turbine colourbar
    from matplotlib.ticker import MaxNLocator
    e_cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    if e_cb_label:
        e_cb.set_label("Per-turbine error in AEP / \%")
    ax.set(xlim=(-EXTENT, EXTENT), ylim=(-EXTENT, EXTENT))

    if farm_label is not None:
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.2)
        ax.annotate(farm_label, xy=(0.05,0.05), ha='left', va='bottom',color='black',xycoords='axes fraction',rotation='horizontal',bbox=props,fontsize=7)

    return cf

def ill_cb(gs,cont_lim):
    cax = fig.add_subplot(gs)
    #cax.invert_xaxis()

    import matplotlib as mpl
    cmap = cmap1
    vmin,vmax = cont_lim
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax, orientation='horizontal',label='Average Wind Velocity / $ms^{-1}$')
    
    return None

from sklearn.metrics.pairwise import euclidean_distances
def find_m_nearest_dists(layout):
    distances = euclidean_distances(layout,layout)
    distances[distances<0.1]=np.nan
    return np.mean(np.nanmin(distances,axis=1))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

wspace = 0.2
gs = GridSpec(12, 4, height_ratios=[14,14,1,1,1,1,1,1,1,8,8,8],wspace=0.2,hspace=0.15)
fig = plt.figure(figsize=(7.8,8), dpi=300) #figsize=(7.8,8)

cont_lim = (np.min(Uwff),np.max(Uwff))

dc = 0 #data column
#this iterates over the columns first then the rows 
for i in range(3): #for each COLUMN
    #first row is the wind roses

    farm_label = wr_label = None
    if FIG_VAR == 1:
        wr_label = f"$U_{{0,i}}={U_AVs[i]}ms^{{-1}}$"
    elif FIG_VAR == 2:
        farm_label = f"{widths[i]/7:.0f}D Eq. Spacing"
    elif FIG_VAR ==3:
        farm_label = f"{len(layouts[i])} Turbines"

    y1 = U_i[:,i]*P_i[:,i]
    
    nice_polar_plot(fig,gs[0,i+1],np.deg2rad(thetaD_WB_i),y1,"$P(\\theta)U(\\theta)$",wr_label=wr_label)
    #next is the contourf
    Z2 = pce(powj_b[i][dc], powj_d[i][dc]) #this is changed!
    xt,yt = layout[i][dc][:,0],layout[i][dc][:,1]
    if i == 1:
        e_cb_label = True
        ff_cb_label = True
    else:
        e_cb_label = False
        ff_cb_label = False
    
    cf = nice_composite_plot_v03B(fig,gs[1,i+1],gs[3,i+1],gs[7,i+1],Uwff[i][dc].reshape(X.shape),X,Y,Z2,xt,yt,errors,cont_lim=cont_lim,ff_cb_label=ff_cb_label,e_cb_label=e_cb_label,farm_label=farm_label)
    
#ill_cb(gs[7,1:],cont_lim) #'illustrative' colourbar on bottom row

aep_a,aep_b,aep_c,aep_d,aep_g = [np.zeros((ROWS,COLS)) for _ in range(5)]
for i in range(ROWS):
    for j in range(COLS):
        aep_a[i,j] = np.sum(powj_a[i][j])
        aep_b[i,j] = np.sum(powj_b[i][j])
        aep_c[i,j] = np.sum(powj_c[i][j])
        aep_d[i,j] = np.sum(powj_d[i][j])
        aep_g[i,j] = np.sum(powj_g[i][j])

aep_arr = np.dstack([aep_b,aep_d])

list_x = [1.2,1.2,1.2,1]
colWidths =  [_/4.6 for _ in list_x]

#AEP table
aep_table_ax = fig.add_subplot(gs[9,:])
aep_table_ax.axis('off')
hdr_list = ['Numerical Integration','GaussianFLOWERS']
aep_row_txt = []
aep_table_text = [['\\textbf{AEP}','','','']]
for i in range(len(hdr_list)): #for each row
    aep_row_txt.append(hdr_list[i]) #first is the name
    for j in range(ROWS): #next 3 are AEP
        aep_row_txt.append(f'{8.760*aep_arr[j,dc,i]:.2f}GWh({pce(aep_arr[j,dc,0],aep_arr[j,dc,i]):+.1f}\%)')
                    
    aep_table_text.append(aep_row_txt)
    aep_row_txt = [] #clear row    

aep_table_ax.table(cellText=aep_table_text, loc='center',colWidths=colWidths,cellLoc='left',edges='open')

time_arr = np.dstack([time_b,time_d,time_g])

# #performance table
# prf_table_ax = fig.add_subplot(gs[10,:])
# prf_table_ax.axis('off')
# hdr_list = ['Numerical Integration','GaussianFLOWERS','JensenFLOWERS']
# prf_row_hdr = []
# prf_table_text = [['\\textbf{Performance}','','','']]
# for i in range(len(hdr_list)): #for each row
#     prf_row_hdr.append(hdr_list[i]) #first is the name
#     for j in range(ROWS): #next 3 are data
#         prf_row_hdr.append(f'{si_fm(time_arr[j,dc,i])}s ({time_arr[j,dc,0]/time_arr[j,dc,i]:.1f})')
                    
#     prf_table_text.append(prf_row_hdr)
#     prf_row_hdr = [] #clear row    
# prf_table_ax.table(cellText=prf_table_text, loc='center',colWidths=colWidths,cellLoc='left',edges='open')

#Error contibution table
err_table_ax = fig.add_subplot(gs[10,:])
err_table_ax.axis('off')
#hdr_list = ['Thrust Coeff','Power Coeff','Cross Terms','Small Angle']
hdr_list = ['$C_t(U_w)\\approx \\overline{{C_t}}$','$C_p(U_w)\\approx C_p(U_\\infty)$','$(\sum x)^N \\approx \sum (x^N)$','Sml Angle','Total']
err_row_hdr = []
err_table_text = [['\\textbf{Error Contributions}','','','']]
for i in range(4): #for each row
    err_row_hdr.append(hdr_list[i]) #first is the name
    for j in range(3): #next 3 are data
        err_row_hdr.append(f'{errors[j,dc,i]:+.1f}\%')                   
    err_table_text.append(err_row_hdr)
    err_row_hdr = [] #clear row    
err_table_ax.table(cellText=err_table_text, loc='center',colWidths=colWidths,cellLoc='left',edges='open')

if SAVE_FIG:
    site_str = ''.join(str(x) for x in site_var)
    layout_str = ''.join(str(x) for x in turb_n)    

    from pathlib import Path
    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "extra evaluations"
    fig_name = f"Fig_FarmEval_withErrors_{site_str}_{layout_str}_P2.png"
    fig_name = "Fig_FarmEval_withErrors"
    path_plus_name = fig_dir / fig_name
    
    plt.savefig(path_plus_name, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")
    print(f"to {path_plus_name}")


#%% "meta" infomration about the wind roses used
idx = 1
u,p = U_i[:,idx],P_i[:,idx]
a = np.sum(u*turb.Cp_f(u)*p*u**3/np.sum(turb.Cp_f(u)*p*u**3))

print(a)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5),dpi=200)
n = np.arange(1,72+1,1)
ax.plot(n,u)
ax.plot(n,p)


#%% "meta" information about the layout used
from sklearn.metrics.pairwise import euclidean_distances

for i in range(3):
    layout = layouts[i]
    distances = euclidean_distances(layout,layout)
    distances[distances<0.1]=np.nan #remove distance from point to itsel
    m_nearest_dists= np.mean(np.nanmin(distances,axis=1))
    print(f"N_turbs ={len(layout)},m_nearest_dists:{m_nearest_dists:.2f}")


#%%
xs = np.linspace(0,30,100)
y1 = turb.Ct_f(xs)
y2 = turb.Cp_f(xs)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5),dpi=200)
ax.plot(xs,y1)
ax.plot(xs,y2)
