#% A figure to show the effect of neglecting the cross terms in the product of the sums
#%% get data to plot
%load_ext autoreload
%autoreload 2
import numpy as np
SAVE_FIG = False
SPACING = 7
INVLD_R = 3
RESOLUTION = 100
EXTENT = 30

def rectangular_domain(extent,r=200):
    X,Y = np.meshgrid(np.linspace(-extent,extent,r),np.linspace(-extent,extent,r))
    return X,Y,np.column_stack((X.reshape(-1),Y.reshape(-1)))

def rectangular_layout(no_xt,s,rot=0):
    n = no_xt//2
    xt = np.arange(-n,n+1,1)*s
    yt = np.arange(-n,n+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)

    phi = np.deg2rad(rot)
    rot_Xt = Xt * np.cos(phi) - Yt * np.sin(phi)
    rot_Yt = Xt * np.sin(phi) + Yt * np.cos(phi)

    return np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))#just a single layout for now

def random_layout(sl,s):
    #sl is an empircal side length
    from AEP3_functions_v01 import poisson2dRandomPoints
    np.random.seed(12341234)
    layouts,distances = poisson2dRandomPoints(1,sl,sl,min_spacing=s,k=30)
    layout = layouts[0,:,:]
    layout = layout[~np.isnan(layout).any(axis=1)]
    layout[:,0] = layout[:,0]-np.mean(layout[:,0]) #center at 0
    layout[:,1] = layout[:,1]-np.mean(layout[:,1])
    return layout

def get_custom_layouts(s):
    layout_array = [[0 for j in range(cols)] for i in range(rows)]
    np.random.seed(12341234)
    for i in range(rows):
        layout_array[i][0] = rectangular_layout(3,s,rot=0)
        layout_array[i][1] = rectangular_layout(5,s,rot=0)
        layout_array[i][2] = rectangular_layout(7,s,rot=0)
        # layout_array[i][2] = random_layout(25,4.9)
        # np.random.seed(12341234)
        # layout_array[i][3] = random_layout(49,6.16)
    return layout_array

def get_wind_roses(no_bins):
    #hard-coded wind roses (this was to allow custom ones...)
    wr_list = []
    for i in range(rows): #for each row
        if i == 0: 
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=2,Cp_f=turb.Cp_f)
        if i == 1: 
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=3,Cp_f=turb.Cp_f)
        if i == 2: 
            wr = wind_rose(bin_no_bins=no_bins,custom=None,site=6,Cp_f=turb.Cp_f)
        wr_list.append(wr)
    return wr_list

def F(u_lim=None,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True):
    #function to show the difference caused by neglecting the cross terms. note: this is NOT the same as cubeAv_v4. The power coefficient is approximated using (directional dependant) global inflow.
    #i:directions,j:turbines,k:turbines in superposistion
    #invalid: specific an invalid radius
    #cross_t: cross terms in cubic expansion
    #sml_a: small_angle approximation
    #local_cp:local power coeff (or global)
    #(var_ct: ct is fixed externally with a lambda function if wanted)
    def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
        
        if u_lim is not None:
            lim = u_lim
        else:
            lim = (np.sqrt(Ct/8)-ep)/K
            lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be <0 but there is numerical artifacting in rsin(theta) )
        
        theta = theta + np.pi #the wake lies opposite!
        if ex: #use full 
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*np.cos(theta)+ep)**2))))*(np.exp(-(r*np.sin(theta))**2/(2*(K*r*np.cos(theta)+ep)**2)))
            deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit  
        else: #otherwise use small angle approximations
            theta = np.mod(theta-np.pi,2*np.pi)-np.pi
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*1+ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+ep)**2)))          
            deltaU_by_Uinf = np.where(r>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
            return deltaU_by_Uinf      
        
        return deltaU_by_Uinf

    def soat(a): #Sum over Axis Two
        return np.sum(a,axis=2)
    
    if var_Ct: 
        Ct_f = turb.Ct_f
    else: #otherwise use the Fixed (Weight averaged) Ct?
        WAV_CT = np.sum(turb.Ct_f(U_i)*P_i)
        Ct_f = lambda x: WAV_CT

    #when plot_points = layout it finds wake at the turbine posistions
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None]

    r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)
    ct_ijk = Ct_f(U_i)[...,None,None]*np.ones((r_jk.shape[0],r_jk.shape[1]))[None,...] #this is a dirty way of repeating along 2 axis

    DU_by_Uinf_ijk = deltaU_by_Uinf_f(r_ijk,theta_ijk,ct_ijk,K) #deltaU_by_Uinf as a function
    if cross_ts: #INcluding cross terms
        Uw_ij_cube = (U_i[:,None]*(1-np.sum(DU_by_Uinf_ijk,axis=2)))**3
    else: #EXcluding cross terms (soat = Sum over Axis Two (third axis!)
        Uw_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(DU_by_Uinf_ijk) + 3*soat(DU_by_Uinf_ijk**2) - soat(DU_by_Uinf_ijk**3))

    Uw_ij = (U_i[:,None]*(1-np.sum(DU_by_Uinf_ijk,axis=2)))
    if lcl_Cp: #power coeff based on local wake velocity
        Cp_ij = Cp_f(Uw_ij)
    else: #power coeff based on global inflow U_infty
        Cp_ij = Cp_f(U_i)[:,None]

    #sum over wind directions (i) (this is the weight-averaging)
    if avCube: #directly find the average of the cube velocity
        pow_j = 0.5*A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uw_ij_cube),axis=0)/(1*10**6)
    else: #the old way of cubing the weight-averaged field
        WAV_CP = np.sum(Cp_f(U_i)*P_i) #frequency-weighted av Cp on global
        pow_j = 0.5*A*RHO*WAV_CP*np.sum(P_i[:,None]*Uw_ij**3,axis=0)/(1*10**6)
 
    return pow_j #power of each turbine/aep flow field in mw

def empty2dPyarray(rows,cols): #create empty 2d python array
    return [[0 for j in range(cols)] for i in range(rows)]

K = 0.025
NO_BINS = 72 #number of bins in the wind rose
RHO = 1.225

rows = 3
cols = 3

from distributions_vC05 import wind_rose 
from AEP3_2_functions import y_5MW
turb = y_5MW()
A = turb.A
wr_list = get_wind_roses(NO_BINS)
X,Y,plot_points = rectangular_domain(EXTENT,r=RESOLUTION)

def gen_local_grid_v02(layout,plot_points):

    xt_j,yt_j = layout[:,0],layout[:,1]
    xt_k,yt_k = plot_points[:,0],plot_points[:,1]

    x_jk = xt_k[:, None] - xt_j[None, :]
    y_jk = yt_k[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    theta_jk = np.pi/2 - np.arctan2(y_jk, x_jk)
    #this is an unneecessary conversion (!)

    return r_jk,theta_jk

theta_i = np.linspace(0,2*np.pi,NO_BINS,endpoint=False)

ff_pce_array = empty2dPyarray(rows,cols) #2d python array
pow_a_arr = empty2dPyarray(rows,cols) 
pow_b_arr = empty2dPyarray(rows,cols) 
a_ff_array = empty2dPyarray(rows,cols) 
#b_ff_array = empty2dPyarray(rows,cols)

layout_array = get_custom_layouts(SPACING)

#generate the contourf data
def pce(exact,approx):
    return 100*(exact-approx)/exact

layout_rot = [0,10,20]
if INVLD_R is not None:
    print(f'overridden invalid zone = {INVLD_R}!')

for i in range(rows):
    wr = wr_list[i] #for each wind rose
    U_i,P_i = wr.avMagnitude,wr.frequency
    Ct_f = turb.Ct_f
    WAV_CT = np.sum(turb.Ct_f(wr.avMagnitude)*wr.frequency)
    fixed_Ct_f = lambda x: WAV_CT
    Cp_f = turb.Cp_f
    for j in range(cols):
        #get all the variables
        layout = layout_array[i][j]
        r_jk,theta_jk = gen_local_grid_v02(layout,layout)
        a_pow = F(u_lim=INVLD_R,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)
        b_pow = F(u_lim=INVLD_R,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=False)
        #then the flow field(the visualisation)
        r_jk,theta_jk = gen_local_grid_v02(layout,plot_points)
        a_ff = F(u_lim=INVLD_R,cross_ts=True,ex=True,lcl_Cp=True,avCube=True,var_Ct=True)

        layout_array[i][j] = layout
        pow_a_arr[i][j] = a_pow
        pow_b_arr[i][j] = b_pow
        a_ff_array[i][j] = a_ff.reshape(X.shape)

        print(f"{cols*i+(j+1)}/{rows*cols}",end="\r")

#%%plot the data ...
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

def nice_polar_plot(fig,gs,x,y,text):
    ax = fig.add_subplot(gs,projection='polar')
    ax.bar(x,y,color='grey',linewidth=1,width=2*np.pi/72)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(60)  # Move radial labels away from plotted line
    ax.text(0, 0, text, ha='left',transform=ax.transAxes,color='black')
    ax.spines['polar'].set_visible(False)

    return None

def nice_polar_plot_v2(gs):#
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  

    def radial_label_botch1():
        labels = ax.get_yticks()
        ax.annotate("$P(\\theta)U(\\theta)$", xy=(0.32,0.8), ha='center', va='center',color='grey',bbox=props,fontsize=6,xycoords='axes fraction',rotation='vertical')
        
        for i in range(len(labels)):
            annotation_txt = f'${labels[i]:.2f}$'  
            ax.annotate(annotation_txt, xy=(0,labels[i]), ha='right', va='top',color='grey',bbox=props,fontsize=6)
            
        return None
    
    def radial_label_botch2():
        labels = ax.get_yticks()
        props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  
        ax.annotate("$C_t(U_\infty)$", xy=(0.8,0.42), ha='center', va='center',color='black',bbox=props,fontsize=6,xycoords='axes fraction')
        
        for i in range(len(labels)):
            annotation_txt = f'${labels[i]/scaler2:.1f}$'  
            ax.annotate(annotation_txt, xy=(np.pi/2,labels[i]), ha='center', va='top',color='black',bbox=props,fontsize=6)
        
        return None
    
    ax = fig.add_subplot(gs,projection='polar')
    theta_i,P_i,U_i = np.deg2rad(wr.deg_bin_centers), wr.frequency,wr.avMagnitude
    y1 = P_i*U_i
    ax.bar(theta_i,y1,color='grey',linewidth=1,width=2*np.pi/72)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    from matplotlib.ticker import FixedLocator,MaxNLocator
    ax.xaxis.set_major_locator(FixedLocator(np.deg2rad([0,90,180,270])))
    ax.xaxis.set_tick_params(pad=0,color='grey',grid_alpha=0.5)
    ax.set_xticklabels(['N', ' ', ' ', ' '])
    ax.spines['polar'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3+1,prune='both'))
    #ax.yaxis.set_tick_params(labelsize='large',color='red')
    ax.yaxis.set_tick_params(pad=0,color='grey',grid_alpha=0.5)
    #the above is bugged, so I have to botch
    ax.set_yticklabels([])
    radial_label_botch1()
    
    labels = ax.get_yticks()
    y2 = turb.Ct_f(U_i)
    scaler2 = labels[-1]/np.max(y2)
    y3 = np.sum(turb.Ct_f(U_i)*P_i)

    print(labels)
    ax.plot(theta_i,scaler2*y2,linewidth=1,color='black')
    radial_label_botch2()    
    print(scaler2)
    ax.plot(theta_i,scaler2*y3*np.ones_like(theta_i),linewidth=1,color='black',ls='--',label='$\overline{C_t}=$'+f'{y3:.2f}')
    ax.legend(loc='upper right',fontsize=5,frameon=False,bbox_to_anchor=(1.0, 1.05),bbox_transform=ax.transAxes)

    return None

from matplotlib import cm
def nice_composite_plot_v02(fig,gs,row,col,Z1,X,Y,Z2,xt,yt,tpce):
    ax = fig.add_subplot(gs[row,col])

    xticks = ax.xaxis.get_major_ticks()
    xticks[2].set_visible(False)
    ax.set_xlabel('$x/d_0$',labelpad=-9)

    yticks = ax.yaxis.get_major_ticks()
    yticks[2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-15)
    ax.tick_params(axis='both', which='both', pad=0,length=3)
    ax.tick_params(axis='x', which='both', pad=2)
    #contourf
    cf = ax.contourf(X,Y,Z1,cmap=cm.gray,levels=50)
    #scatter plot
    color_list = plt.cm.coolwarm(np.linspace(0, 1, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(color_list)
    sp = ax.scatter(xt,yt,c=Z2,cmap=cmap,marker='x',s=10,vmin=0)
    cax = fig.add_subplot(gs[row+1,col])
    cb = fig.colorbar(sp, cax=cax, orientation='horizontal',format='%.3g')
    from matplotlib.ticker import MaxNLocator
    cb.ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set(xlim=(-EXTENT, EXTENT), ylim=(-EXTENT, EXTENT))
    # cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')

    annotation_txt = f'${tpce:+.2f}$\\%'  
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  
    ax.annotate(annotation_txt, xy=(0.97,0.97), ha='right', va='top',color='black',bbox=props,xycoords='axes fraction',fontsize=9)

    if row == 4 and col == 2:
        cax.set_xlabel('Percentage Error in AEP / \%',labelpad=2)
        
    return cf

def ill_cb(): #the illustrative colourbar
    cax = fig.add_subplot(gs[7,1:])
    cax.invert_xaxis()
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
    from matplotlib.ticker import FixedLocator, FixedFormatter
    cb.ax.xaxis.set_major_locator(FixedLocator([np.max(a_ff), np.min(a_ff)]))  # Set tick positions
    cb.ax.xaxis.set_major_formatter(FixedFormatter(['less waked', 'more waked']))  # Set tick labels
    
    return None


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
br = 30
gs = GridSpec(8, cols+1, height_ratios=[br,1,br,1,br,1,1,1],wspace=0.15,hspace=0.41)
fig = plt.figure(figsize=(7.8,8), dpi=300)

import numpy as np

for i in range(rows): 
    #first column is the wind rose
    wr = wr_list[i]
    wind_rose_y = wr.frequency*wr.avMagnitude
    nice_polar_plot_v2(gs[2*i,0])
    for j in range(cols): #then onto the contours
        Z2 = pce(pow_a_arr[i][j], pow_b_arr[i][j])
        tpce = pce(np.sum(pow_a_arr[i][j]),np.sum(pow_b_arr[i][j])) 
        xt,yt = layout_array[i][j][:,0],layout_array[i][j][:,1]
        cf = nice_composite_plot_v02(fig,gs,2*i,j+1,a_ff_array[i][j],X,Y,Z2,xt,yt,tpce) 

ill_cb() #'illustrative' colourbar on bottom row

#%%
if SAVE_FIG:
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches='tight')

    print("figure saved")

#%%
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(1.7,1.7), dpi=200)
wr = wr_list[1]

def nice_polar_plot_v2():#
    props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  

    def radial_label_botch1():
        labels = ax.get_yticks()
        ax.annotate("$P(\\theta)U(\\theta)$", xy=(0.32,0.8), ha='center', va='center',color='grey',bbox=props,fontsize=5,xycoords='axes fraction',rotation='vertical')
        
        for i in range(len(labels)):
            annotation_txt = f'${labels[i]:.2f}$'  
            ax.annotate(annotation_txt, xy=(0,labels[i]), ha='right', va='top',color='grey',bbox=props,fontsize=5)
            
        return None
    
    def radial_label_botch2():
        labels = ax.get_yticks()
        props = dict(boxstyle='round', facecolor='white', alpha=0.8,edgecolor='none',pad=0.1)  
        ax.annotate("$C_t(U_\infty)$", xy=(0.8,0.42), ha='center', va='center',color='black',bbox=props,fontsize=5,xycoords='axes fraction')
        
        for i in range(len(labels)):
            annotation_txt = f'${labels[i]/scaler2:.1f}$'  
            ax.annotate(annotation_txt, xy=(np.pi/2,labels[i]), ha='center', va='top',color='black',bbox=props,fontsize=5)
        
        return None
    
    ax = fig.add_subplot(projection='polar')
    theta_i,P_i,U_i = np.deg2rad(wr.deg_bin_centers), wr.frequency,wr.avMagnitude
    y1 = P_i*U_i
    ax.bar(theta_i,y1,color='grey',linewidth=1,width=2*np.pi/72)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    from matplotlib.ticker import FixedLocator,MaxNLocator
    ax.xaxis.set_major_locator(FixedLocator(np.deg2rad([0,90,180,270])))
    ax.xaxis.set_tick_params(pad=0,color='grey',grid_alpha=0.5)
    ax.set_xticklabels(['N', ' ', ' ', ' '])
    ax.spines['polar'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3+1,prune='both'))
    #ax.yaxis.set_tick_params(labelsize='large',color='red')
    ax.yaxis.set_tick_params(pad=0,color='grey',grid_alpha=0.5)
    #the above is bugged, so I have to botch
    ax.set_yticklabels([])
    radial_label_botch1()
    
    labels = ax.get_yticks()
    y2 = turb.Ct_f(U_i)
    scaler2 = labels[-1]/np.max(y2)
    y3 = np.sum(turb.Ct_f(U_i)*P_i)

    ax.plot(theta_i,scaler2*y2,linewidth=1,color='black')
    radial_label_botch2()    

    ax.plot(theta_i,scaler2*y3*np.ones_like(theta_i),linewidth=1,color='black',ls='--',label='$\overline{C_t}=$'+f'{y3:.2f}')
    ax.legend(loc='upper right',fontsize=5,frameon=False,bbox_to_anchor=(1, 1),bbox_transform=ax.transAxes)

    return None

nice_polar_plot_v2()
plt.show()

