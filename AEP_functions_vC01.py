
import numpy as np
from scipy import integrate
from timer import stopwatch

#USE AN EXACT FUNCTION?
EXACT = False
class num_functions():

    def a_exact(self,r,theta,params):
        #exact gaussian
        k,ep,Ct,X_LIM,r_lim = params  
        DU_w = (1-np.sqrt(1-(Ct/(8*(k*r*np.cos(theta)+ep)**2))))*np.exp(-0.5*((r*np.sin(theta))/(k*r*np.cos(theta)+ep))**2)
        DU_w = np.where(r*np.cos(theta)>X_LIM,DU_w,0) #NaNs mess with sum later
        DU_w = np.where(r>6,DU_w,0) #user defined r limit
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

    def gen_local_grid(self,plot_grid,co_ords):
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

    def ni_average_cubed(self,rs,thetas,params,cjd_f,feedback=True):
        #\overline{U}^3 by numerical quadrature    
        def b(thetaPrime,r,theta,params):
            sum = np.sum(self.a(r,theta-thetaPrime,params)) #wake velocity deficit sum
            return (1-sum)*cjd_f(thetaPrime)
        
        thetas = thetas+np.pi # wake lies opposite
        
        c = np.full_like(rs[:,0],0)
        errs = np.zeros_like(c)

        loops = np.shape(rs)[0]
        timer = stopwatch() 
        for i in range(loops):
            r = rs[i,:] #for each local r 
            theta = thetas[i,:] # and theta
            #perform the numerical integration
            c[i],errs[i] = integrate.quad(b,-np.pi, np.pi,args=(r,theta,params))
            #produce nice feedback
            if feedback==True and i%(loops//10)==0 and i != 0:#feedback every ~10%                  
                timer.lap(i/loops,"UA3: ")    
        return c**3, errs 

    #numerical integration cubed average
    def ni_cubed_average(self,rs,thetas,params,cjd3_f,feedback=False):
        #\overline{U^3} by numerical quadrature    
        
        def b(thetaPrime,r,theta,params):
            sum = np.sum(self.a(r,theta-thetaPrime,params)) #wake velocity deficit sum
            return ((1-sum)**3)*(cjd3_f(thetaPrime))

        thetas = thetas+np.pi # wake lies opposite

        c = np.full_like(rs[:,0],0)
        errs = np.zeros_like(c)

        loops = np.shape(rs)[0]
        timer = stopwatch() 
        for i in range(np.shape(rs)[0]): #for each (global) r and theta
            r = rs[i,:] #for each local r 
            theta = thetas[i,:] # and theta
            #perform the numerical integration
            c[i],errs[i] = integrate.quad(b,-np.pi, np.pi,args=(r,theta,params)) 
            #produce nice feedback
            if feedback==True and i%(loops//10)==0 and i != 0:#feedback every ~10%                  
                timer.lap(i/loops,"UA3: ")    

        return c, errs 

    def discrete_convolution(self,r,theta,djd,f):
        #convolves a discrete sequence djd with a function f
        no_bins = np.size(djd)
        bin_thetas = np.linspace(0,2*np.pi,no_bins,endpoint=False)
        
        #the shift arrays for the sum
        r_c = np.repeat(r[:,:,None],no_bins,axis=2) 
        theta_c = theta[:,:,None]-bin_thetas + np.pi 

        fs = f(r_c,theta_c) #take samples using function

        return np.sum(djd[None,:]*fs,axis=1) #sum over theta shift

    def nc_average_cubed(self,rs,thetas,params,djd):
        #\overline{U}^3 by numerical convolution 

        def b(r,theta):
            return 1-np.sum(self.a(r,theta,params),axis=1)
        
        U_w = self.discrete_convolution(rs,thetas,djd,b)
        return U_w**3

    def nc_cubed_average(self,rs,thetas,params,djd3):
        #\overline{U^3} by numerical convolution 

        def b(r,theta):
            return (1-np.sum(self.a(r,theta,params),axis=1))**3
        
        U_w = self.discrete_convolution(rs,thetas,djd3,b)
        return U_w

    #hh this bit could be drastically simplified
    #but maybe this is easier to understand
    def A_full(self,rs,thetas,params,djd3):

        def b(r,theta):
            return +1
        
        aa = self.discrete_convolution(rs,thetas,djd3,b)
        return +1*aa

    def B_full(self,rs,thetas,params,djd3):

        def b(r,theta):
            return np.sum(self.a(r,theta,params),axis=1)
        
        bb = self.discrete_convolution(rs,thetas,djd3,b)
        return -3*bb

    def C_full(self,rs,thetas,params,djd3):

        def b(r,theta):
            return np.sum(self.a(r,theta,params),axis=1)**2
        
        cc = self.discrete_convolution(rs,thetas,djd3,b)
        return +3*cc

    def discrete_convolution_2(self,r,theta,djd,f):
        #slightly different function
        #takes the sum AFTER convolving (so no cross terms)
        no_bins = np.size(djd)
        bin_thetas = np.linspace(0,2*np.pi,no_bins,endpoint=False)
        #the shift arrays for the sum
        r_c = np.repeat(r[:,:,None],no_bins,axis=2) 
        theta_c = theta[:,:,None]-bin_thetas + np.pi 
        fs = f(r_c,theta_c) #take samples using function
        return np.sum(djd[None,None,:]*fs,axis=2) #sum over theta shift 

    def C_simp(self,rs,thetas,params,djd3):

        def b(r,theta):
            return self.a(r,theta,params)**2
        
        cc_simp = self.discrete_convolution_2(rs,thetas,djd3,b)

        return +3*np.sum(cc_simp,axis=1)

    def D_full(self,rs,thetas,params,djd3):

        def b(r,theta):
            return np.sum(self.a(r,theta,params),axis=1)**3
        
        dd = self.discrete_convolution(rs,thetas,djd3,b)
        return -dd

    def D_simp(self,rs,thetas,params,djd3):

        def b(r,theta):
            return self.a(r,theta,params)**3
        
        dd_simp = self.discrete_convolution_2(rs,thetas,djd3,b)
        return -1*np.sum(dd_simp,axis=1)
            

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
def poisson2dRandomPoints(no_grids,width,height,min_spacing,k=30):
    #Generate a """randomly*""" (*not entirely random) spread set of points in a 2d plane
    #plane_of_coords is a list of a list of tuples (simple.... right?). 
    #avNearestNeighbour is the average distance to the nearest neighbour and is a numpy array
    avNearestNeighbour = np.zeros((no_grids))
    plane_of_coords  = []
    for grid in range(no_grids):
        object = PoissonDisc(width=width,height=height,r=min_spacing,k=k)
        co_ords = object.sample()
        plane_of_coords.append(co_ords)
        distances = euclidean_distances(co_ords,co_ords)
        distances[distances<min_spacing]=np.NaN #remove distance from point to itself
        avNearestNeighbour[grid] = np.mean(np.nanmin(distances,axis=1))
    return plane_of_coords, avNearestNeighbour
