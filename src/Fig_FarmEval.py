#%% this is the 3x3 figure comparing the ntag to the floris implemented method

#%% get data to plot
import sys
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2

import numpy as np
SAVE_FIG = False
EXTRA_INFO = False
SPACING = 7 #turbine spacing normalised by rotor DIAMETER
U_LIM = 3 #manually override ("user limit") the invalid radius around the turbine (otherwise variable, depending on k/Ct) - 
RESOLUTION = 100 #number of x/y points in contourf meshgrid
EXTENT = 30 #total size of contourf "window" (square from -EXTENT,-EXTENT to EXTENT,EXTENT)
K = 0.03 #expansion parameter for the Gaussian model
NO_BINS = 72 #number of bins in the wind rose
ROWS = 3 #number of sites
COLS = 3 #number of layout variations

def empty2dPyarray(rows,cols): #create empty 2d python array
    return [[0 for j in range(cols)] for i in range(rows)]

def find_errors(U_i,P_i,theta_i,layout,plot_points,turb,K):
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
                     layout,
                     plot_points,
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

theta_i = np.linspace(0,360,NO_BINS,endpoint=False) 

from utilities.turbines import iea_10MW
turb = iea_10MW()

site_n = [2,3,6] #[6,8,10] are also tricky 
layout_n = [5,6,7] # update EXTENT to increase size of window if increasing this
rot = [0,0,0]
#generate the contourf data
from utilities.helpers import simple_Fourier_coeffs,get_floris_wind_rose,get_WAV_pp,rectangular_layout,fixed_rectangular_domain,adaptive_timeit

X,Y,plot_points = fixed_rectangular_domain(EXTENT,r=RESOLUTION)

layout,powj_a,powj_b,powj_c,powj_d,powj_e,powj_f= [empty2dPyarray(ROWS, COLS) for _ in range(7)]  #2d python arrays
time_a,time_c,time_d,time_e = [np.zeros((ROWS,COLS)) for _ in range(4)]

Uwff_b = np.zeros((ROWS,COLS,plot_points.shape[0]))

U_i,P_i = [np.zeros((NO_BINS,len(site_n))) for _ in range(2)]

errors = np.zeros((ROWS,COLS,4))

from utilities.AEP3_functions import floris_AV_timed_aep,num_Fs,vect_num_F,ntag_PA,caag_PA

for i in range(ROWS): #for each wind rose (site)
    U_i[:,i],P_i[:,i] = get_floris_wind_rose(site_n[i])
    #For ntag, the fourier coeffs are found from Cp(Ui)*Pi*Ui**3
    _,Fourier_coeffs3_PA = simple_Fourier_coeffs(turb.Cp_f(U_i[:,i])*(P_i[:,i]*(U_i[:,i]**3)*len(P_i[:,i]))/(2*np.pi))
    #For caag, the fourier coeffs are found from Pi*Ui
    _,Fourier_coeffs_noCp_PA = simple_Fourier_coeffs((P_i[:,i]*U_i[:,i]*len(P_i[:,i]))/(2*np.pi))
    
    wav_Ct = get_WAV_pp(U_i[:,i],P_i[:,i],turb,turb.Ct_f) #weight ct by power production

    for j in range(COLS): #for each layout
        timed = True #timing toggle
        layout[i][j] = rectangular_layout(layout_n[j],SPACING,rot[j])
        
        #find the errors due to each assumption (numerically)
        errors[i,j,:] = find_errors(U_i[:,i],P_i[:,i],np.deg2rad(theta_i),layout[i][j],plot_points,turb,K)

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

        #vectorised numerical aep (aep+time)
        aep_func_c = lambda: vect_num_F(U_i[:,i],P_i[:,i],np.deg2rad(theta_i),
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
    
        # #floris NO WAKE aep
        powj_f[i][j],_ = floris_AV_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,wake=False,timed=False)   
        
        print(f"{COLS*i+(j+1)}/{ROWS*COLS}\r")

#%%plot the data ...
from utilities.plotting_funcs import set_latex_font
set_latex_font() #set latex font 

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

from utilities.plotting_funcs import si_fm
from utilities.helpers import pce

def nice_composite_plot_v03(fig,gs,i,j,Z1,X,Y,Z2,xt,yt,errors,cont_lim=(None,None)):
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

    aep_a = np.sum(powj_a[i][j]) #SOA reference
    aep_d = np.sum(powj_d[i][j]) #proposed model

    top_left_text = f'''{aep_a:.2f}MW(ref) in {si_fm(time_a[i][j])}s
    {aep_d:.2f}MW({pce(aep_a,aep_d):+.1f}\%) in {si_fm(time_d[i][j])}s({time_a[i][j]/time_d[i][j]:.2f})'''

    ax.text(0.05,0.95,top_left_text,color='black',transform=ax.transAxes,va='top',ha='left',fontsize=4,bbox=props)
    if EXTRA_INFO: #extra info that is discussed
        error_text = f'''$C_t$:{errors[0]:+.1f}
        $C_p$:{errors[1]:+.1f}
        Xt1:{errors[2]:+.1f}
        SA:{errors[3]:+.1f}
        Sum:{np.sum(errors[:4]):+.1f}
        '''
        ax.text(0.05,0.05,error_text,color='black',transform=ax.transAxes,va='bottom',ha='left',fontsize=4,bbox=props)

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
        cf = nice_composite_plot_v03(fig,gs,i,j,Uwff_b[i][j].reshape(X.shape),X,Y,Z2,xt,yt,errors[i,j,:],cont_lim=cont_lim) 

ill_cb(gs[6,1:],cont_lim) #'illustrative' colourbar on bottom row

if SAVE_FIG:
    site_str = ''.join(str(x) for x in site_n)
    layout_str = ''.join(str(x) for x in layout_n)    

    from pathlib import Path

    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "fig images"
    fig_name = f"Fig_FarmEval_{site_str}_{layout_str}.png"
    path_plus_name = fig_dir / fig_name
    
    plt.savefig(path_plus_name, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")

plt.show()
