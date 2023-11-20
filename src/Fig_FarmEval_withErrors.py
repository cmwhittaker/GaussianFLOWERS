#%% Produce the 3x3 wind roses(sites)xlayouts figure that compares the accuracy of Gaussian FLOWERS to a floris reference
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

#I used ipy and don't want to fat finger and wait 20 min for it to run again
run = False
SAVE_FIG = False
timed = True 

if not run:
    raise ValueError('This cell takes a long time to run - are you sure you meant to run this cell?')

import numpy as np

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

layout,powj_a,powj_b,powj_c,powj_d,powj_e,powj_f,powj_g,powj_h= [empty2dPyarray(ROWS, COLS) for _ in range(9)]  #2d python arrays
time_a,time_c,time_d,time_e,time_g,time_h = [np.zeros((ROWS,COLS)) for _ in range(6)]

Uwff_b = np.zeros((ROWS,COLS,plot_points.shape[0]))

U_i,P_i = [np.zeros((NO_BINS,len(site_n))) for _ in range(2)]

errors = np.zeros((ROWS,COLS,4))

from utilities.AEP3_functions import floris_AV_timed_aep,num_Fs,vect_num_F,ntag_PA,caag_PA,flowers_timed_aep,floris_FULL_timed_aep

for i in range(ROWS): #for each wind rose (site)
    U_i[:,i],P_i[:,i],fl_wr = get_floris_wind_rose(site_n[i])
    #For ntag, the fourier coeffs are found from Cp(Ui)*Pi*Ui**3
    _,Fourier_coeffs3_PA = simple_Fourier_coeffs(turb.Cp_f(U_i[:,i])*(P_i[:,i]*(U_i[:,i]**3)*len(P_i[:,i]))/(2*np.pi))
    #For caag, the fourier coeffs are found from Pi*Ui
    _,Fourier_coeffs_noCp_PA = simple_Fourier_coeffs((P_i[:,i]*U_i[:,i]*len(P_i[:,i]))/(2*np.pi))
    
    wav_Ct = get_WAV_pp(U_i[:,i],P_i[:,i],turb,turb.Ct_f) #weight ct by power production

    for j in range(COLS): #for each layout
        
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

        #flowers AEP
        powj_g[i][j],time_g[i][j] = flowers_timed_aep(U_i[:,i],P_i[:,i],theta_i,layout[i][j],turb,0.05,timed=timed)
        
        #floris AEP WITHOUT wind speed averaging
        powj_h[i][j],time_h[i][j] = floris_FULL_timed_aep(fl_wr,theta_i,layout[i][j],turb,timed=False)

        print(f"{COLS*i+(j+1)}/{ROWS*COLS}\r")

#%% simple no wake sanity check
n,m = 2,2
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
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.2)
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
    cb = fig.colorbar(sp, cax=cax, cmap=cmap,orientation='horizontal',format='%.3g') #the per turbine colourbar
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set(xlim=(-EXTENT, EXTENT), ylim=(-EXTENT, EXTENT))
    if i==2 and j==1: #middle bottom label the colourbar
        cb.set_label("Per-turbine error in AEP / \%")

    #Then the farm total values:
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,pad=0.2)

    aep_a = np.sum(powj_a[i][j]) #floris reference
    aep_b = np.sum(powj_b[i][j]) #analytical AEP directly
    aep_c = np.sum(powj_c[i][j]) #AEP from cubed weight average velocity
    aep_d = np.sum(powj_d[i][j]) 
    aep_e = np.sum(powj_e[i][j]) 
    aep_f = np.sum(powj_f[i][j]) 
    aep_g = np.sum(powj_g[i][j]) 
    aep_h = np.sum(powj_h[i][j]) 

    aep_ref = aep_h
    time_ref = time_c[i][j]
                        
    top_left_text = f'''CumCurl\hspace{{2.8ex}}:{aep_h:.2f}MW({pce(aep_ref,aep_h):+.1f}\%) 
    CumCurlav:{aep_a:.2f}MW({pce(aep_ref,aep_a):+.1f}\%) in {si_fm(time_a[i][j])}s({time_ref/time_a[i][j]:.2f}) 
    NumInt*\hspace{{3ex}}:{aep_c:.2f}MW({pce(aep_ref,aep_c):+.1f}\%) in {si_fm(time_c[i][j])}s({time_ref/time_c[i][j]:.2f})
    GFlowers\hspace{{3ex}}:{aep_d:.2f}MW({pce(aep_ref,aep_d):+.1f}\%) in {si_fm(time_d[i][j])}s({time_ref/time_d[i][j]:.2f})
    NoWake\hspace{{4ex}}:{aep_f:.2f}MW({pce(aep_ref,aep_f):+.1f}\%)
    JFlowers\hspace{{3.8ex}}:{aep_g:.2f}MW({pce(aep_ref,aep_g):+.1f}\%) in {si_fm(time_g[i][j])}s({time_ref/time_g[i][j]:.2f})'''

    #NumInt\hspace{{4.6ex}}:{aep_b:.2f}MW({pce(aep_a,aep_b):+.1f}\%)
    #GFlowers**:{aep_e:.2f}MW({pce(aep_a,aep_e):+.1f}\%) in {si_fm(time_e[i][j])}s({time_a[i][j]/time_e[i][j]:.2f})

    ax.text(0.05,0.95,top_left_text,color='black',transform=ax.transAxes,va='top',ha='left',fontsize=4,bbox=props)

    error_text = f'''Error Contributions
    $C_t(U_w)\\approx \\overline{{C_t}}$:{errors[0]:+.1f}\%
    $C_p(U_w)\\approx C_p(U_\\infty)$:{errors[1]:+.1f}\%
    $(\sum x)^N \\approx \sum (x^N)$:{errors[2]:+.1f}\%
    Sml Angle:{errors[3]:+.1f}\%
    Total:{np.sum(errors[:4]):+.1f}\%
    '''
    ax.text(0.05,0.05,error_text,color='black',transform=ax.transAxes,va='bottom',ha='left',fontsize=4,bbox=props)

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
gs = GridSpec(8, 4, height_ratios=[14,1,14,1,14,1,1,1],wspace=0.3,hspace=0.41)
fig = plt.figure(figsize=(10,10), dpi=300) #figsize=(7.8,8)

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

ill_cb(gs[7,1:],cont_lim) #'illustrative' colourbar on bottom row

if SAVE_FIG:
    site_str = ''.join(str(x) for x in site_n)
    layout_str = ''.join(str(x) for x in layout_n)    

    from pathlib import Path

    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "extra evaluations"
    fig_name = f"Fig_FarmEval_withErrors_{site_str}_{layout_str}.png"
    path_plus_name = fig_dir / fig_name
    
    plt.savefig(path_plus_name, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")

plt.show()

#%% Another ... version of the figure

def nice_composite_plot_v03B(fig,cf_gs,cb_gs,Z1,X,Y,Z2,xt,yt,errors,cont_lim=(None,None),cb_label=False):
    ax = fig.add_subplot(cf_gs)

    xticks = ax.xaxis.get_major_ticks()
    xticks[2].set_visible(False)
    ax.set_xlabel('$x/d_0$',labelpad=-9)

    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-19)
    ax.yaxis.set_tick_params(pad=0)
    #contourf
    vmin,vmax = cont_lim
    cf = ax.contourf(X,Y,Z1,50,cmap=cmap1,vmin=vmin,vmax=vmax)
    #scatter plot
    color_list = plt.cm.coolwarm(np.linspace(0, 1, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(color_list)
    sp = ax.scatter(xt,yt,c=Z2,cmap=cmap,marker='x',s=10)
    #sp = ax.scatter(xt,yt,marker='x',s=10,c='black')
    cax = fig.add_subplot(cb_gs)
    cb = fig.colorbar(sp, cax=cax, cmap=cmap,orientation='horizontal',format='%.3g') #the per turbine colourbar
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    if cb_label:
        cb.set_label("Per-turbine error in AEP / \%")
    ax.set(xlim=(-EXTENT, EXTENT), ylim=(-EXTENT, EXTENT))

    return cf

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

wspace = 0.2
gs = GridSpec(12, 4, height_ratios=[14,14,1,1,1,1,1,1,1,12,12,12],wspace=0.2,hspace=0.15)
fig = plt.figure(figsize=(7.8,8.9), dpi=300) #figsize=(7.8,8)

cont_lim = (np.min(Uwff_b),np.max(Uwff_b))

dc = 2 #data column
#this iterates over the columns first then the rows 
for i in range(3): #for each COLUMN
    #first row is the wind roses
    y1 = U_i[:,i]*P_i[:,i]
    nice_polar_plot(fig,gs[0,i+1],np.deg2rad(theta_i),y1,"$P(\\theta)U(\\theta)$")
    #next is the contourf
    Z2 = pce(powj_a[i][dc], powj_d[i][dc])
    xt,yt = layout[i][dc][:,0],layout[i][dc][:,1]
    if i == 1:
        cb_label = True
    else:
        cb_label = False
    cf = nice_composite_plot_v03B(fig,gs[1,i+1],gs[3,i+1],Uwff_b[i][dc].reshape(X.shape),X,Y,Z2,xt,yt,errors,cont_lim=cont_lim,cb_label=cb_label)
    
ill_cb(gs[7,1:],cont_lim) #'illustrative' colourbar on bottom row

aep_a,aep_c,aep_d,aep_g = [np.zeros((3,3)) for _ in range(4)]
for i in range(3):
    for j in range(3):
        aep_a[i,j] = np.sum(powj_a[i][j])
        aep_c[i,j] = np.sum(powj_c[i][j])
        aep_d[i,j] = np.sum(powj_d[i][j])
        aep_g[i,j] = np.sum(powj_g[i][j])

aep_arr = np.dstack([aep_a, aep_c,aep_d,aep_g])

list_x = [1.2,1.2,1.2,1]
colWidths =  [_/4.6 for _ in list_x]

#AEP table
aep_table_ax = fig.add_subplot(gs[9,:])
aep_table_ax.axis('off')
hdr_list = ['CumulativeCurl','Numerical Integration','GaussianFLOWERS','JensenFLOWERS']
aep_row_txt = []
aep_table_text = [['\\textbf{AEP}','','','']]
for i in range(4): #for each row
    aep_row_txt.append(hdr_list[i]) #first is the name
    for j in range(3): #next 3 are AEP
        aep_row_txt.append(f'{aep_arr[j,dc,i]:.2f}MW({pce(aep_arr[j,dc,0],aep_arr[j,dc,i]):+.1f}\%)')
                    
    aep_table_text.append(aep_row_txt)
    aep_row_txt = [] #clear row    

aep_table_ax.table(cellText=aep_table_text, loc='center',colWidths=colWidths,cellLoc='left',edges='open')

time_arr = np.dstack([time_a,time_c,time_d,time_g])

#performance table
prf_table_ax = fig.add_subplot(gs[10,:])
prf_table_ax.axis('off')
hdr_list = ['CumulativeCurl','Numerical Integration','GaussianFLOWERS','JensenFLOWERS']
prf_row_hdr = []
prf_table_text = [['\\textbf{Performance}','','','']]
for i in range(4): #for each row
    prf_row_hdr.append(hdr_list[i]) #first is the name
    for j in range(3): #next 3 are data
        prf_row_hdr.append(f'{si_fm(time_arr[j,dc,i])}s ({time_arr[j,dc,0]/time_arr[j,dc,i]:.1f})')
                    
    prf_table_text.append(prf_row_hdr)
    prf_row_hdr = [] #clear row    
prf_table_ax.table(cellText=prf_table_text, loc='center',colWidths=colWidths,cellLoc='left',edges='open')

#Error contibution table
err_table_ax = fig.add_subplot(gs[11,:])
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
    site_str = ''.join(str(x) for x in site_n)
    layout_str = ''.join(str(x) for x in layout_n)    

    from pathlib import Path
    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "extra evaluations"
    fig_name = f"Fig_FarmEval_withErrors_{site_str}_{layout_str}_P2.png"
    path_plus_name = fig_dir / fig_name
    
    plt.savefig(path_plus_name, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")

