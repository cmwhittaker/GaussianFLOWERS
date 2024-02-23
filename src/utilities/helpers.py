#%% random helper functions
import numpy as np
def deltaU_by_Uinf_f(r,theta,Ct,K,u_lim,ex):
    '''
    wake velocity deficit based on Bastankah 2014
    Args: (normalised by DIAMETER)
        r (np.array): r in polar coordinates
        theta (np.array): theta in polar coordinates
        Ct (np.array | float): Thrust coefficient
        K (np.array | float): Wake expansion coefficient
        u_lim (float): user defined invalid radius limit. This sets a radius around the turbine where the deficit is zero (useful for plotting)
        ex (bool): Choice to use exact wake deficit formula (True) or a small angle approximated (False)
    Returns:
        deltaU_by_Uinf (np.array): wake velocity fraction at (r,theta)
    
    '''
    ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct))) #initial expansion width: (eq.6 + 19 +21 in Bastankah 2014 - don't forget eq21!)
    if u_lim != None: #override the limit with the user defined radius
        lim = u_lim
    else:
        lim = (np.sqrt(Ct/8)-ep)/K #invalid region
        lim = np.where(lim<0.1,0.1,lim) #may sure it's always atleast 0.01 (stop self-produced wake) 
    if ex: #use full 
        U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*np.cos(theta)+ep)**2))))*(np.exp(-(r*np.sin(theta))**2/(2*(K*r*np.cos(theta)+ep)**2)))
        deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit  
    else: #otherwise use small angle approximations
        theta = np.mod(theta-np.pi,2*np.pi)-np.pi #fix domain
        U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*1+ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+ep)**2)))          
        deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
        return deltaU_by_Uinf      
    
    return deltaU_by_Uinf  

def linear_layout(no_xt,s):
    #returns a linear 1 x no_xt turbine grid layout starting at  (0,0) then continuing at (s,0),(2*s,0) ... etc
    xt = np.arange(0,no_xt*s,s)
    yt = np.zeros_like(xt)
    return xt,yt,np.column_stack((xt,yt))

def rectangular_layout(no_xt,s,rot):
    #returns a rectangular no_xt x no_xt turbine grid layout centered on (0,0) with clockwise rotation (in radians!) rot 
    low = (no_xt)/2-0.5
    xt = np.arange(-low,low+1,1)*s
    yt = np.arange(-low,low+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)
    Xt,Yt = [_.reshape(-1) for _ in [Xt,Yt]]
    rot_Xt = Xt * np.cos(rot) + Yt * np.sin(rot)
    rot_Yt = -Xt * np.sin(rot) + Yt * np.cos(rot) 
    layout = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
    return layout#just a single layout for now

def rectangular_domain(layout,xpad=7,ypad=7,xr=100,yr=100): 
    '''
    Returns a rectilinear grid of points shape (xr*yr,2) given a layout.
    The grid extends by xpad and ypad padding each side, with number of points xr and yr. (Normalised by rotor diameter)
    '''
    xmin,xmax = np.min(layout[:,0]),np.max(layout[:,0])
    ymin,ymax = np.min(layout[:,1]),np.max(layout[:,1])
    xlims = (xmin-xpad,xmax+xpad)
    ylims = (ymin-ypad,ymax+ypad)
    x = np.linspace(xlims[0],xlims[1],xr)
    y = np.linspace(ylims[0],ylims[1],yr)
    xx,yy = np.meshgrid(x,y)
    return xx,yy,np.column_stack((xx.reshape(-1),yy.reshape(-1))),xlims,ylims

def fixed_rectangular_domain(extent,r=200):
    #rectilinear grid shape (r**2,2) over rectangle centered on (0,0) with side lengths 2*extent
    xx,yy = np.meshgrid(np.linspace(-extent,extent,r),np.linspace(-extent,extent,r))
    return xx,yy,np.column_stack((xx.reshape(-1),yy.reshape(-1)))

def find_relative_coords(plot_points,layout):
    #find the r, theta coordinates relative to each turbine
    #when plot_points is a meshgrid (stacked as coords), then this can be used to visualise the wake nicely
    xt_n,yt_n = plot_points[:,0],plot_points[:,1]
    xt_m,yt_m = layout[:,0],layout[:,1]
    #relative cartesian
    x_nm = xt_n[:, None] - xt_m[None,:] #dimensions (n,m)
    y_nm = yt_n[:, None] - yt_m[None,:] #as written in paper
    #convert to polar
    r_nm = np.sqrt(x_nm**2+y_nm**2)
    theta_nm = np.arctan2(y_nm,x_nm) # (clockwise -ve from x axis
    # remove diagonal elements
    # (I don't do this because it messes with visualisation ... )
    # mask = ~np.eye(n, dtype=bool)
    # out = x3[:,mask].reshape(m,n,n-1)

    return r_nm,theta_nm  

def simple_Fourier_coeffs(data):   
    # naively fit a Fourier series to data (no normalisation takes place (!))
    # returns both sine/cosine and phase/amplitude coefficients
    # reconstruction uses the formula:
    # a_0/2 + (a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)
    # or 
    # a_0/2 + (A_n*np.cos(n_b*theta_b+Phi_n)
    import scipy.fft
    c = scipy.fft.rfft(data)/np.size(data)
    a_0 = 2*np.real(c[0]) 
    a_n = 2*np.real(c[1:])
    b_n =-2*np.imag(c[1:])
    Fourier_coeffs = a_0,a_n,b_n
    # #convert to phase amplitude form
    A_n = np.sqrt(a_n**2+b_n**2)
    Phi_n = -np.arctan2(b_n,a_n)
    #add the dc term on the front with Phi_0 = 0
    A_n = np.concatenate((np.array((a_0/2,)),A_n))
    Phi_n = np.concatenate((np.array((0,)),Phi_n))
    Fourier_coeffs_PA = A_n,Phi_n #pack
    
    return Fourier_coeffs,Fourier_coeffs_PA

def get_WAV_pp(U_i,P_i,turb,f):
    #use power production to weight-average function f
    #(there may be better ways)
    WAV = np.sum(f(U_i)*turb.Cp_f(U_i)*P_i*U_i**3/np.sum(turb.Cp_f(U_i)*P_i*U_i**3))
    return WAV

def get_WAV_pr(U_i,P_i,f):
    #use probability to weight average function f
    #(there may be better ways)
    WAV = np.sum(f(U_i)*P_i)
    return WAV

def trans_bearing_to_polar(U_WB_i,P_WB_i,theta_WB_i):
    #converts wind bearing theta_WB_i to polar angle theta_i
    #fixes domain, and then re-sorts U_WB_i and P_WB_i to match
    theta_i = (3*np.pi)/2 - theta_WB_i#convert to polar
    theta_i = np.mod(theta_i,2*np.pi) #fix from 0 to 2pi

    srt_idx = np.argsort(theta_i) #re-sort using transformed theta
    theta_i = theta_i[srt_idx]
    U_i = U_WB_i[srt_idx]
    P_i = P_WB_i[srt_idx]


    return U_i,P_i,theta_i

from floris.tools import WindRose
def get_floris_wind_rose(site_n,align_west=False,**kwargs):
    #use floris to parse wind rose toolkit site data
    #(each site has its own folder)
    from pathlib import Path
    current_file_path = Path(__file__)
    folder_name = current_file_path.parent.parent.parent/ "data" / "WindRoseData_D" / ("site"+str(site_n))

    fl_wr = WindRose()
    fl_wr.parse_wind_toolkit_folder(folder_name,limit_month=None,**kwargs)
    wr = fl_wr.resample_average_ws_by_wd(fl_wr.df)
    wr.freq_val = wr.freq_val/np.sum(wr.freq_val) #frequency sumf to 1
    if align_west: #align predominant direction West
        offsets = [-70,210,-35,250,250,270,190,-35,215,245,265,145]
        wr.wd = wr.wd - offsets[site_n-1] - 90

    U_WB_i = wr.ws #(average) wind speeds
    P_WB_i = wr.freq_val 
    theta_WB_i = np.deg2rad(wr.wd) #wind direction bearing (in radians)
    #convert wind bearings to polar angle (and re-sort, U_i)
    U_i,P_i,theta_i = trans_bearing_to_polar(U_WB_i,P_WB_i,theta_WB_i)
    
    return np.array(U_i),np.array(P_i),np.array(theta_i),fl_wr

def get_floris_wind_rose_WB(site_n,**kwargs):
    #use floris to parse wind rose toolkit site data
    #(each site has its own folder)
    from pathlib import Path
    current_file_path = Path(__file__)
    folder_name = current_file_path.parent.parent.parent/ "data" / "WindRoseData_D" / ("site"+str(site_n))

    fl_wr = WindRose()
    fl_wr.parse_wind_toolkit_folder(folder_name,limit_month=None,**kwargs)
    wr = fl_wr.resample_average_ws_by_wd(fl_wr.df)
    wr.freq_val = wr.freq_val/np.sum(wr.freq_val)
    U_WB_i = wr.ws #(average) wind speeds
    P_WB_i = wr.freq_val 
    theta_WB_i = np.deg2rad(wr.wd) #wind direction bearing (in radians)
    
    return np.array(U_WB_i),np.array(P_WB_i),np.array(theta_WB_i),fl_wr

#signed percentage error
def pce(exact,approx):
    return 100*(approx-exact)/exact

def empty2dPyarray(rows,cols): #create empty 2d python array
    return [[0 for j in range(cols)] for i in range(rows)]

def empty3dPyarray(rows,cols,lays): #create empty 3d python array
    return [[[0 for k in range(lays)] for j in range(cols)] for i in range(rows)]

import numpy as np
import timeit
def adaptive_timeit(func,timed=True,repeats = 5,runtime=3):
    result = func() #get the actual result of the function
    if timed is not True: #don't bother timing
        return result,np.NaN
    # determine appropriate number of executions
    for index in range(0, 10):
        number = 10 ** index
        time_number = timeit.timeit(func,number=number)
        if time_number >= runtime/(repeats*10):
            break
    if number == 1: #for long functions
        repeats = 3
    xcts = timeit.repeat(func, number=number, repeat=repeats)
    return result, np.min(xcts)/number 

from scipy.stats import vonmises
def vonMises_wr(Uav,kappa,dir=0,NO_BINS=72):
    # return a wind rose with direction probability defined
    # by the von Mises function
    # wind speed is uniform 
    # dir = 0 by default which is West in the polar coord system
    theta_i = np.linspace(0,2*np.pi,NO_BINS,endpoint=False)
    U_i = Uav*np.ones_like(theta_i)
    P_i = vonmises.pdf(dir, kappa, theta_i)
    P_i = P_i / np.sum(P_i) #normalise to ensure sum to 1
    return U_i,P_i,theta_i

from utilities.poissonDiscSampler import PoissonDisc
def random_layouts(layout_n,width=42,min_r=5.1,k=30):
    #returns list of 2d coord arrays
    layout_list = []
    for i in range(layout_n):
        sampler = PoissonDisc(width,width,min_r,k)
        coords = sampler.sample()
        layout = np.asarray(coords)
        layout = layout - width/2 #center around (0,0)
        layout_list.append(layout)
    return layout_list