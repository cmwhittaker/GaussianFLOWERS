#%% random helper functions
import numpy as np
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

def gen_local_grid(layout,plot_points):
    #find the r, theta coordinates relative to each turbine
    xt_j,yt_j = layout[:,0],layout[:,1]
    xt_k,yt_k = plot_points[:,0],plot_points[:,1]

    x_jk = xt_k[:, None] - xt_j[None, :]
    y_jk = yt_k[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    #convert theta from clckwise -ve x axis to clckwise +ve y axis 
    theta_jk = np.pi/2 - np.arctan2(y_jk, x_jk)

    return r_jk,theta_jk  

def simple_Fourier_coeffs(data):   
    # naively fit a Fourier series to data (no normalisation takes place (!))
    # returns both sine/cosine and phase/amplitude coefficients
    # reconstruction uses the formula:
    # a_0/2 + (a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)
    # or 
    # a_0/2 + (A_n*np.cos(n_b*theta_b+Phi_n)
    import scipy.fft
    c = scipy.fft.rfft(data)/np.size(data)
    length = np.size(c)-1 #because the a_0 term is included in c # !! 
    a_0 = 2*np.real(c[0])
    a_n = 2*np.real(c[-length:])
    b_n =-2*np.imag(c[-length:])
    Fourier_coeffs = a_0,a_n,b_n
    # #convert to phase amplitude form
    A_n = np.sqrt(a_n**2+b_n**2)
    Phi_n = -np.arctan2(b_n,a_n)
    Fourier_coeffs_PA = a_0,A_n,Phi_n
    
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

from floris.tools import WindRose
def get_floris_wind_rose(site_n):
    #use floris to parse wind rose toolkit site data
    #(each site has its own folder)
    fl_wr = WindRose()
    folder_name = "data/WindRoseData_D/site" +str(site_n)
    fl_wr.parse_wind_toolkit_folder(folder_name,limit_month=None)
    wr = fl_wr.resample_average_ws_by_wd(fl_wr.df)
    wr.freq_val = wr.freq_val/np.sum(wr.freq_val)
    U_i = wr.ws
    P_i = wr.freq_val
    return np.array(U_i),np.array(P_i)

#signed percentage error
def pce(exact,approx):
    return 100*(approx-exact)/exact