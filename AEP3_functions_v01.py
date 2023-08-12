#%% This aims to be a "definitive" location for all the functions used in "Phase 3" of the AEP project

import numpy as np
from scipy import integrate
from timer import stopwatch

def gen_local_grid(plot_grid,co_ords):
    #generate a stack of co-ordinates, each plane being local to a single turbine (to later sum over)

    X,Y = np.array(plot_grid)[:,0],np.array(plot_grid)[:,1] #plot co-ords X and Ys
    xt,yt = np.array(co_ords)[:,0],np.array(co_ords)[:,1] #turbine coords x and ys
    #find coordinates local to each turbine posistion
    loc_x = (X[:,None]-xt[None,:]) #(NPlotPoints,NTurbines)
    loc_y = (Y[:,None]-yt[None,:]) #(NPlotPoints,NTurbines)
    #convert to a polar-coordinate system (still local)
    loc_r = np.sqrt(loc_x**2+loc_y**2) 
    loc_theta = np.arctan2(loc_x,loc_y)
    return loc_r,loc_theta

#USE AN EXACT FUNCTION?
class num_functions():

    def a_exact(self,r,theta,params):
        #exact gaussian
        k,ep,Ct,X_LIM,r_lim = params  
        DU_w = (1-np.sqrt(1-(Ct/(8*(k*r*np.cos(theta)+ep)**2))))*np.exp(-0.5*((r*np.sin(theta))/(k*r*np.cos(theta)+ep))**2)
        DU_w = np.where(r*np.cos(theta)>X_LIM,DU_w,0) #NaNs mess with sum later
        DU_w = np.where(r>r_lim,DU_w,0) #user defined r limit
        return DU_w

    def a_approx(self,r,theta,params):
        #approx gaussian
        k,ep,Ct,X_LIM,r_lim = params  
        DU_w = (1-np.sqrt(1-(Ct/(8*(k*r+ep)**2))))*np.exp(-0.5*(r*(np.mod(theta-np.pi,2*np.pi)-np.pi)/(k*r+ep))**2)
        DU_w = np.where(r>r_lim,DU_w,0)
        return DU_w     
    
    def __init__(self,object_name="",exact=True):
        #initalise the functions (choose between approx and exact)
        if exact == True:
            print(object_name+"~EXACT wake function used")
            self.a = self.a_exact
        else:
            print(object_name+"~APPROX wake function used")
            self.a = self.a_approx
        return None

    def discrete_convolution(self,r,theta,djd,f):
        #convolves a discrete sequence djd with a function f
        no_bins = np.size(djd)
        bin_thetas = np.linspace(0,2*np.pi,no_bins,endpoint=False)
        
        #the shift arrays for the sum
        r_c = np.repeat(r[:,:,None],no_bins,axis=2) 
        theta_c = theta[:,:,None]-bin_thetas + np.pi 
        print("r_c.shape: {}".format(r_c.shape))
        fs = f(r_c,theta_c) #take samples using function
        print("(djd[None,:]*fs).shape: {}".format((djd[None,:]*fs).shape))
        return np.sum(djd[None,:]*fs,axis=1) #sum over theta shift

    def nc_cubed_average(self,rs,thetas,params,djd3):
        #cubed average is the one you want
        #\overline{U^3} by numerical convolution 

        def b(r,theta):
            return (1-np.sum(self.a(r,theta,params),axis=1))**3
        
        U_w = self.discrete_convolution(rs,thetas,djd3,b)
        return U_w

    def nc_average_cubed(self,rs,thetas,params,djd):
        #\overline{U}^3 by numerical convolution 

        def b(r,theta):
            return 1-np.sum(self.a(r,theta,params),axis=1)
        
        U_w = self.discrete_convolution(rs,thetas,djd,b)
        return U_w**3

class num_functions_v2():
    #version 2 now lets Ct vary over the wind direction
    #(based on the mean wind speed of that bin)
    def a_exact(self,r,theta,params):
        #exact gaussian
        #params is now an array so it needs unpacking
        Ct,ep,X_LIM,r_lim = np.split(params,params.shape[-1],axis=-1)
        Ct,ep,X_LIM,r_lim = Ct[...,0],ep[...,0],X_LIM[...,0],r_lim[...,0]
        DU_w = (1-np.sqrt(1-(Ct/(8*(self.k*r*np.cos(theta)+ep)**2))))*np.exp(-0.5*((r*np.sin(theta))/(self.k*r*np.cos(theta)+ep))**2)
        DU_w = np.where(r*np.cos(theta)>X_LIM,DU_w,0) #NaNs mess with sum later
        DU_w = np.where(r>r_lim,DU_w,0) #user defined r limit
        return DU_w
    
    def re_calculate_constants2(self,u):
        ct= self.Ct_f(u)
        ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct))) 
        x_lim = (np.sqrt(ct/8)-ep)/self.k
        r_lim = (np.sqrt(ct/8)-ep)/self.k
        params = np.column_stack((ct,ep,x_lim,r_lim))
        return params

    def __init__(self,Ct_f,k=0.03):
        #k is the wake growth parameter
        #Ct_f is the thrust coefficient interpolation function
        print("EXACT wake function used")
        print("k: {}".format(k))
        self.a = self.a_exact
        self.k = k
        self.Ct_f = Ct_f
        return None

    def nc_cubed_average(self,rs,thetas,djd3,av_u):
        #cubed average is the one you want
        #\overline{U^3} by numerical convolution 
        no_bins = np.size(djd3)
        bin_thetas = np.linspace(0,2*np.pi,no_bins,endpoint=False)
        
        #the shift arrays for the sum
        r_c = np.repeat(rs[:,:,None],no_bins,axis=2) 
        theta_c = thetas[:,:,None]-bin_thetas + np.pi 
        #the parameters (Ct) varies with wind direction
        param_array = self.re_calculate_constants2(av_u)

        #calculate the samples of the wake
        s = (1-np.sum(self.a(r_c,theta_c,param_array[None,None,:,:]),axis=1) )**3 #sum over the turbines
        #convolves a discrete sequence djd with SAMPLES s
        #s is the same length as djd
        conv_out = np.sum(s*djd3[None,:],axis=1)
        return conv_out #sum over theta shift

def ctag(rs,thetas,PARAMS,fourier_terms):
    #"cross term analytical Gaussian" - the performance can probably be increased with a small amount of optimisation

    thetas = thetas - np.pi #wake is opposite to the incoming wind
    a_0,a_n,b_n = fourier_terms
    K,EP,CT, X_LIM, R_LIM = PARAMS
    #auxilaries
    n = np.arange(1,a_n.size+1,1)
    sigma = np.where(rs!=0,(K*rs+EP)/rs,0)
    sqrt_term = np.where(rs<(np.sqrt(CT/8)-EP)/K,0,(1-np.sqrt(1-(CT/(8*(K*rs+EP)**2)))))

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]    
    sigma_b = sigma[:,:,None]
    a_n = a_n[None,None,:]
    b_n = b_n[None,None,:]
    theta_b = thetas[:,:,None]

    def term(a):
        cnst_term = (np.sqrt(2*np.pi*a)*sigma/(2*a*np.pi))*(sqrt_term**a)
        mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2*a))*(a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)

    #I don't know why this 2pi is needed, *but it is*
    return (a_0/2 - 3*term(1) + 3*term(2) - term(3))*2*np.pi

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import numpy as np

def nice_contour_plot(params,row,column,data_to_plot,label,cb_label=None):
    
    def close_to_zero(x):
        #stops silly labels like 0.123e-16
        if abs(x)<0.05:
            return 0
        else:
            return x
    
    fig,gs,X,Y,co_ords= params
    #Note: grid-spec is ZERO indexed!
    ax = fig.add_subplot(gs[row,column])
    #contour plot (note:matplotlib doesn't always choose correct levels)
    data_min = close_to_zero(np.min(data_to_plot))
    data_max = close_to_zero(np.max(data_to_plot))
    #print("row: {} column: {} data_min: {} data_max: {}".format(row,column,data_min,data_max))
    if data_min == data_max: #special case for the DC graph
        cf = ax.contourf(X,Y,data_to_plot,10,cmap=cm.coolwarm)
    else: #gets confused on the DC graph
        cf = ax.contourf(X,Y,data_to_plot,np.linspace(data_min,data_max,10),cmap=cm.coolwarm)
    #contour plot axis decorations
    ax.set_xlabel('$x/d_0$',labelpad=-9)
    xticks = ax.xaxis.get_major_ticks()
    xticks[len(xticks)//2].set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[len(yticks)//2].set_visible(False)
    ax.set_ylabel('$y/d_0$',labelpad=-19)
    ax.tick_params(axis='y', which='major', pad=1)
    #colourbar and decorations
    cax = fig.add_subplot(gs[row+1,column])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
    cb.ax.locator_params(nbins=5)
    cb.ax.tick_params(labelsize=8)
    if cb_label != None:
        cax.set_xlabel(cb_label,labelpad=5,size=8)
    #X's on turbine locations
    xt,yt = np.array(co_ords)[:,0],np.array(co_ords)[:,1]
    ax.scatter(xt,yt,marker='x',color='white')
    #label
    ax.text(0.98*np.max(X), 0.98*np.max(Y), label, size=8, ha="right", va="top",bbox=dict(boxstyle="square",ec='w', fc='w',pad=0.15))
    return None

def nice_polar_plot(fig,gs,row,column,f):

    #first column is the wind rose
    ax = fig.add_subplot(gs[row,column],projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    xt=np.linspace(0,2*np.pi,1000)
    ax.plot(xt,f(xt),color='black')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(60)  # Move radial labels away from plotted line
    return None

from poissonDiscSampler import PoissonDisc
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
def poisson2dRandomPoints(no_grids,width,height,min_spacing=3,k=30):
    #Generate a """randomly*""" (*not entirely random) spread set of points in a 2d plane
    #plane_of_coords is a list of a list of tuples (simple.... right?). 
    #avNearestNeighbour is the average distance to the nearest neighbour and is a numpy array
    avNearestNeighbour = np.zeros((no_grids))
    tri_area = (np.sqrt(3)/4)*min_spacing**3
    turbine_estimate = 3*int(((width+min_spacing)*(width+min_spacing))/tri_area)+1
    plane_of_coords  = np.full((no_grids,turbine_estimate,2),np.NaN)

    for grid in range(no_grids):
        object = PoissonDisc(width=width,height=height,r=min_spacing,k=k)
        co_ords = np.array(object.sample())

        plane_of_coords[grid,:co_ords.shape[0],:] = co_ords

        distances = euclidean_distances(co_ords,co_ords)
        distances[distances<min_spacing]=np.NaN #remove distance from point to itself
        avNearestNeighbour[grid] = np.mean(np.nanmin(distances,axis=1))
    return plane_of_coords, avNearestNeighbour

class VestaV80():
    def __init__(self):
        import numpy as np
        self.Cp = np.array([0, 0, 0.420716448, 0.4257312255, 0.4307844413, 0.4361689172, 0.438602936, 0.4442110823, 0.4487245914, 0.4509839599, 0.447164543, 0.441598962, 0.4294423023, 0.4120601569, 0.3900904517, 0.3641078662, 0.3354289493, 0.3042230301, 0.2733876129, 0.2446877071,0.2204419594, 0.1998609572, 0.1811584087, 0.1641863998, 0.149269929, 0.1361069938, 0.1244473091, 0.1140822632, 0.1048370421, 0.09656440951, 0.08913976227, 0.08245717283,0, 0])

        self.Ct = np.array([0, 0, 0.8064186284, 0.80428, 0.80428, 0.80428, 0.8049425473, 0.8060155067, 0.80687, 0.8067478964, 0.8032742313, 0.7982236139, 0.7777002664, 0.7526508598, 0.7422251481, 0.7150607311, 0.6326699895, 0.5131534121, 0.423818614, 0.3592138243, 0.3057028244, 0.2640244532, 0.2270639194, 0.1996650081, 0.1762732959, 0.1580905773, 0.1434216257, 0.1328810227, 0.1242298655, 0.1172546581, 0.1104905987, 0.104873448, 0, 0])

        self.wind_speed = np.array([0, 4.99, 5, 5.612244898, 6.12244898, 6.632653061, 7.142857143, 7.653061224, 8.163265306, 8.673469388, 9.183673469, 9.693877551, 10.20408163, 10.71428571, 11.2244898, 11.73469388, 12.24489796, 12.75510204, 13.26530612, 13.7755102, 14.28571429, 14.79591837, 15.30612245, 15.81632653, 16.32653061, 16.83673469, 17.34693878,17.85714286, 18.36734694, 18.87755102, 19.3877551, 19.99, 20, 50])

        self.U_h = 70 #height
        self.d = 80 #diameter
        self.A = np.pi*(self.d/2)**2 #area

    def Ct_itrp(self,u):
        return np.interp(u,self.wind_speed,self.Ct)

    def Cp_itrp(self,u):
        return np.interp(u,self.wind_speed,self.Cp)  
    
    def Cp_f(self):
        def f(u):
            return np.interp(u,self.wind_speed,self.Cp)
        return f

class y_5MW():
    def __init__(self):
        import numpy as np
        self.Cp = np.array([0. , 0. , 0.074 , 0.3251 , 0.3762 , 0.4027 , 0.4156 , 0.423 , 0.4274 , 0.4293 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.429603, 0.354604, 0.316305, 0.281478, 0.250068, 0.221924, 0.196845, 0.174592, 0.154919, 0.13757 , 0.1223 , 0.108881, 0.097094, 0.086747, 0.077664, 0.069686, 0.062677, 0.056511, 0.051083, 0.046299, 0.043182, 0.033935, 0. , 0. ])

        self.Ct = np.array([0. , 0. , 0.7701, 0.7701, 0.7763, 0.7824, 0.782 , 0.7802, 0.7772, 0.7719, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7675, 0.7651, 0.7587, 0.5056, 0.431 , 0.3708, 0.3209, 0.2788, 0.2432, 0.2128, 0.1868, 0.1645, 0.1454, 0.1289, 0.1147, 0.1024, 0.0918, 0.0825, 0.0745, 0.0675, 0.0613, 0.0559, 0.0512, 0.047 , 0. , 0. ])

        self.wind_speed = np.array([ 0. , 2.9 , 3. , 4. , 4.5147, 5.0008, 5.4574, 5.8833, 6.2777, 6.6397, 6.9684, 7.2632, 7.5234, 7.7484, 7.9377, 8.0909, 8.2077, 8.2877, 8.3308, 8.337 , 8.3678, 8.4356, 8.5401, 8.6812, 8.8585, 9.0717, 9.3202, 9.6035, 9.921 , 10.272 , 10.6557, 11.5077, 12.2677, 12.7441, 13.2494, 13.7824, 14.342 , 14.9269, 15.5359, 16.1675, 16.8204, 17.4932, 18.1842, 18.8921, 19.6152, 20.3519, 21.1006, 21.8596, 22.6273, 23.4019, 24.1817, 24.75 , 25.01 , 25.02 , 50. ])

        self.Z_h = 90 #height
        self.D = 126 #diameter
        self.A = np.pi*(self.D/2)**2 #area

    def Ct_itrp(self,u):
        return np.interp(u,self.wind_speed,self.Ct)

    def Cp_itrp(self,u):
        return np.interp(u,self.wind_speed,self.Cp)  
    
    def Cp_f(self):
        def f(u):
            return np.interp(u,self.wind_speed,self.Cp)
        return f

    def Ct_f(self):
        def f(u):
            return np.interp(u,self.wind_speed,self.Ct)
        return f