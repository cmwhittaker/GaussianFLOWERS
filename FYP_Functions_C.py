#%%All the most recent and efficient wake model programs consolidated into a single place.
#All functions use a North-aligned theta co-ordinate system
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy import integrate

def reconstructFourier(x,fourier_coeffs):
    #reconstruct function from the Fourier coefficients, useful for visualising the wind rose used in the analytical integration
    a_0,a_n,b_n = fourier_coeffs
    n = np.array((np.arange(1,a_n.size+1))) #halve open things
    return a_0 + np.sum(a_n[:,None]*np.cos(n[:,None]*x)+b_n[:,None]*np.sin((n[:,None]*x)),axis=0)

def reconstructFourier_asFunction(fourier_coeffs):
    #reconstruct function from the Fourier coefficients, but return a (continous) function, not just discrete points
    a_0,a_n,b_n = fourier_coeffs
    n = np.array((np.arange(1,a_n.size+1))) #halve open things
    def f(x):
        return a_0 + np.sum(a_n[:,None]*np.cos(n[:,None]*x)+b_n[:,None]*np.sin((n[:,None]*x)),axis=0)
    return f

def binWindData(direction,magnitude,bin_width,graph=False):
    #Sort the wind data into bins by direction, then calculate an average velocity magnitude for each bin.
    #direction in degrees,magnitude in m/s, bin_width in degrees
    #note: The function returns the direction bins in radians!
    if 360%bin_width != 0:
        raise ValueError("The bin width is not a factor of 360")
    no_bins = int(360/bin_width)
    #bins are left/right +0.5/0.5 a bin_width
    bins = np.arange(bin_width/2,360,bin_width)
    bin_centres = bins-bin_width/2
    #index of which bin data belongs to bin_number = [0,0,1,1,2 ...]
    bin_number = np.digitize(direction,bins)
    frequency = np.zeros((no_bins),dtype='float64')
    avmagnitude = np.zeros((no_bins),dtype='float64')
    #The first bin needs to combine first and last bin bc of 2pi periodicy
    frequency[0] = np.size(direction[bin_number==0]) + np.size(direction[bin_number==no_bins])
    avmagnitude[0] = (np.mean(magnitude[bin_number==0]) + np.mean(magnitude[bin_number==no_bins]))/2
    for i in range(1,no_bins):
        frequency[i] = np.size(direction[bin_number==i])
        avmagnitude[i] = np.mean(magnitude[bin_number==i])
    #Finally, express the frequency as a fraction (normalise)
    frequency = frequency/np.sum(frequency)
    #The exact average of the velocity (DC term, a_0) (*possibly misleading-not weighted by P!)
    Xa_0 = np.mean(avmagnitude)
    if graph==True:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.title("frequency")
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.plot(np.deg2rad(bin_centres),frequency,color='green')

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.title("magnitude")
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.plot(np.deg2rad(bin_centres),avmagnitude,color='red')

    return np.deg2rad(bin_centres),frequency,avmagnitude,Xa_0

# from poissonDiscSampler import PoissonDisc
# from sklearn.metrics.pairwise import euclidean_distances
# def poisson2dRandomPoints(no_grids,width,height,min_spacing,k=30):
#     #Generate a """randomly*""" (*not entirely random) spread set of points in a 2d plane
#     #plane_of_coords is a list of a list of tuples (simple.... right?). 
#     #avNearestNeighbour is the average distance to the nearest neighbour and is a numpy array
#     avNearestNeighbour = np.zeros((no_grids))
#     plane_of_coords  = []
#     for grid in range(no_grids):
#         object = PoissonDisc(width=width,height=height,r=min_spacing,k=k)
#         co_ords = object.sample()
#         plane_of_coords.append(co_ords)
#         distances = euclidean_distances(co_ords,co_ords)
#         distances[distances<min_spacing]=np.NaN #remove distance from point to itself
#         avNearestNeighbour[grid] = np.mean(np.nanmin(distances,axis=1))
#     return plane_of_coords, avNearestNeighbour

def ac(r,theta,params,jd_fourier_coeffs,r_min=None,invalid=0):
    #analytical convolution (with the small angle Gaussian model)
    #r: (n,) - the radii to evaluate the convolution at
    #theta: (n,) - the thetas to evaluate the convolution at
    #fourier_coeffs: (1d,list of list,list of list) - Fourier coeffs of the WR_PDF
    #RETURNS:
    #DU_w: (n,) - the wake velocity deficit at (r,theta)
    a_0,a_n,b_n=jd_fourier_coeffs #unpack Fourier coefficients
    a_n,b_n = a_n.reshape(-1),b_n.reshape(-1) #reshape them to 1D
    k,ep,Ct = params #unpack model parameters
    if r_min == None:
        r_min = (np.sqrt(Ct/8)-ep)/k #r limit for invalid region
    theta = theta + np.pi #wake lies opposite
    n = np.arange(1,np.size(a_n)+1,1) #n:(N_fourier_terms,) - ns in the sigma
    sigma_a = (k*r+ep)/r #The standard deviation approximation
    sum_exp_terms= np.exp((-0.5*sigma_a[:,None]**2)*n[None,:]**2) 
    sum_trig_terms = a_n[None,:]*np.cos(n[None,:]*theta[:,None])+b_n[None,:]*np.sin(n[None,:]*theta[:,None]) #Fourier terms
    DU_w = sigma_a*(np.sqrt(2*np.pi))*(1-np.sqrt(1-(Ct/(8*(k*r+ep)**2))))*(a_0+np.sum(sum_exp_terms*sum_trig_terms,1))
    DU_w = np.where(r>r_min,DU_w,invalid) #remove the invalid zone close to turb
    return DU_w 

def ac_nt(plot_grid,co_ords,params,jd_fourier_coeffs,r_min=None,invalid=0):
    #analytical convolution, n turbine (with the small angle Gaussian model)
    #will find the wake velocity at the turbine posistion when plot_grid == co_ords 
    #plot_grid: (2,n) - the co-ords of points to evaluate the velocity deficit at
    #co_ords: (2,m) - the co-ords of the turbines
    #RETURNS:
    #DU_w: (2,n) - the wake velocity deficit at the plot points
    X,Y = np.array(plot_grid)[:,0],np.array(plot_grid)[:,1] #plot co-ords X and Ys
    xt,yt = np.array(co_ords)[:,0],np.array(co_ords)[:,1] #turbine coords x and ys
    #find coordinates local to each turbine posistion
    loc_x = (X[:,None]-xt[None,:]) #(NPlotPoints,NTurbines)
    loc_y = (Y[:,None]-yt[None,:]) #(NPlotPoints,NTurbines)
    #convert to a polar-coordinate system (still local)
    loc_r = np.sqrt(loc_x**2+loc_y**2).reshape(-1) #reshape to pass to ac()
    loc_theta = np.arctan2(loc_x,loc_y).reshape(-1) 
    #calculate every point in local co-ordinate 'space'
    loc_DU_w = ac(loc_r,loc_theta,params,jd_fourier_coeffs,r_min=r_min,invalid=invalid)
    #reshape to form a stack of 2d planes for each local coordinate grid,then sum 
    DU_w = np.sum(loc_DU_w.reshape(loc_x.shape),axis=1)
    return DU_w

def ej_sd(r,theta,params,x_lim=0,invalid=0):
    #exact Jensen, single direction
    k,ep,Ct = params 
    rcos = r*np.cos(theta) #(x)
    rsin = r*np.sin(theta) #(y)
    b = np.where((abs(rsin)<k*rcos+0.5)&(rcos>x_lim))
    DU_w = np.full_like(rcos,invalid)
    DU_w[b] = (1-np.sqrt(1-Ct))/((1+(2*k*rcos[b]))**2)
    return DU_w

def eg_sd(r,theta,params,x_lim,invalid=0):
    #exact Gaussian, single direction
    #when called by nc, this calculation is quite sparse, so it is worth the overhead to first find what needs calculating and only calculate what is needed.
    k,ep,Ct = params  
    rsin = r*np.sin(theta)
    rcos = r*np.cos(theta)
    bracket = (k*rcos+ep)**2
    DU_w = np.full_like(rcos,invalid)
    b = np.where(rcos>x_lim)
    DU_w[b] = (1-np.sqrt(1-(Ct/(8*bracket[b]))))*np.exp((-0.5*(rsin[b])**2)/bracket[b])
    return DU_w

def nc(r,theta,params,djd,r_lim=None,invalid=0,mode='gauss'):
    #Numerical convolution of the single-direction wake velocity deficit with the wind rose
    k,ep,Ct = params #unpack model parameters
    x_lim = (np.sqrt(Ct/8)-ep)/k #the x invalid zone limit
    if r_lim == None:
        r_lim = x_lim
    no_bins = np.size(djd)
    bin_thetas = np.linspace(0,2*np.pi,no_bins,endpoint=False) #points to sample the WR at
    r1 = np.repeat(r[:,None],no_bins,axis=1) #convolving over constant r
    theta1 = theta[:,None]-bin_thetas + np.pi #the shift array for the sum
    if mode=='gauss':
        Uw_sd = eg_sd(r1,theta1,params,x_lim,invalid)
        Uw_sd = np.where(r1>r_lim,Uw_sd,invalid)
    elif mode=='jensen':
        Uw_sd = ej_sd(r1,theta1,params,x_lim,invalid)
    else:
        raise ValueError('Invalid mode selected :(, should be either "gauss" or "jensen"')

    Uw_md = np.sum(djd[None,:]*Uw_sd,axis=1)
    return Uw_md

def nc_nt(plot_grid,co_ords,params,djd,r_lim=None,mode='gauss'):
    #WRONG WRONG WRONG !!!!!!!! Neeeds to be (U_w - dU_w)^3 CAREFUL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #numerical convolution, n turbine (with the small angle Gaussian model)
    #will find the wake velocity at the turbine posistion when plot_grid == co_ords 
    #plot_grid: (2,n) - the co-ords (cartesian!) of points to evaluate the velocity deficit at
    #co_ords: (2,m) - the co-ords (cartesian!) of the turbines
    #RETURNS:
    #DU_w: (2,n) - the wake velocity deficit at the plot points
    X,Y = np.array(plot_grid)[:,0],np.array(plot_grid)[:,1] #plot co-ords X and Ys
    xt,yt = np.array(co_ords)[:,0],np.array(co_ords)[:,1] #turbine coords x and ys
    #find coordinates local to each turbine posistion
    loc_x = (X[:,None]-xt[None,:]) #(NPlotPoints,NTurbines)
    loc_y = (Y[:,None]-yt[None,:]) #(NPlotPoints,NTurbines)
    #convert to a polar-coordinate system (still local)
    loc_r = np.sqrt(loc_x**2+loc_y**2).reshape(-1) #reshape to pass to nc()
    loc_theta = np.arctan2(loc_x,loc_y).reshape(-1) 
    #calculate every point in local co-ordinate 'space'
    loc_DU_w = nc(loc_r,loc_theta,params,djd,r_lim=r_lim,mode=mode)
    #reshape to form a stack of 2d planes for each local coordinate grid,then sum 
    DU_w = np.sum(loc_DU_w.reshape(loc_x.shape),axis=1)
    return DU_w



def convf(thetaPrime,r,theta,params,x_lim,jd_func):
    #A function used in the "exact" integration convolution function
    #Careful! This is not exactly the function within the convolution integral, adding pi is done externally to increase efficiency
    k,ep,Ct = params  
    with np.errstate(invalid='ignore'): #turn off annoying warning
        f = (1-np.sqrt(1-(Ct/(8*(k*r*np.cos(thetaPrime)+ep)**2))))*np.exp((-0.5*(r*np.sin(thetaPrime))**2)/(k*r*np.cos(thetaPrime)+ep)**2)
    f = np.where(r*np.cos(thetaPrime)>x_lim,f,0) # restrict the domain
    g = jd_func(theta-thetaPrime)
    return f*g

from scipy import integrate
def ei(rs,thetas,params,jd_func,x_min=None,invalid=0,feedback=False):
    thetas = thetas+np.pi # wake lies opposite
    k,ep,Ct = params #unpack model parameters
    if x_min == None:
        x_lim = (np.sqrt(Ct/8)-ep)/k #r limit for invalid region    
    DU_ws = np.full_like(rs,invalid)
    errs = np.zeros_like(rs)
    it = np.nditer([rs,thetas],flags=['c_index']) #np iterator
    for r,theta in it:
        i = it.index
        if r<x_lim:
            DU_ws[i],errs[i] = invalid,np.NaN #in the invalid zone, don't bother calculating
        else:
            DU_ws[i],errs[i] = integrate.quad(convf,-np.pi, np.pi,args=(r,theta,params,x_lim,jd_func)) #record the wake velocity deficit and the error estimate
        if feedback==True and i%(np.size(rs)/10)==0:#feedback every ~10%
            print("{}/{}".format(i+np.size(rs)/10,np.size(rs)))
    return DU_ws, errs 

def ei_nt(plot_grid,co_ords,params,jd_func,x_min=None,invalid=0,feedback=False):
    #"exact" integration, n turbine (with the exact Gaussian model)
    #will find the wake velocity at the turbine posistion when plot_grid == co_ords 
    #plot_grid: (2,n) - the co-ords (cartesian!) of points to evaluate the velocity deficit at
    #co_ords: (2,m) - the co-ords (cartesian!) of the turbines
    #RETURNS:
    #DU_w: (2,n) - the wake velocity deficit at the plot points
    X,Y = np.array(plot_grid)[:,0],np.array(plot_grid)[:,1] #plot co-ords X and Ys
    xt,yt = np.array(co_ords)[:,0],np.array(co_ords)[:,1] #turbine coords x and ys
    #find coordinates local to each turbine posistion
    loc_x = (X[:,None]-xt[None,:]) #(NPlotPoints,NTurbines)
    loc_y = (Y[:,None]-yt[None,:]) #(NPlotPoints,NTurbines)
    #convert to a polar-coordinate system (still local)
    loc_r = np.sqrt(loc_x**2+loc_y**2).reshape(-1) #reshape to pass to ac()
    loc_theta = np.arctan2(loc_x,loc_y).reshape(-1) 
    #calculate every point in local co-ordinate 'space'
    loc_DU_w,err = ei(loc_r,loc_theta,params,jd_func,x_min=x_min,invalid=invalid,feedback=feedback)
    #reshape to form a stack of 2d planes for each local coordinate grid,then sum 
    DU_w = np.sum(loc_DU_w.reshape(loc_x.shape),axis=1)
    err.reshape(loc_x.shape)
    return DU_w,err

def fourier_from_csi_pdf(csi_pdf,terms,a0=1):
    #DEPRECIATED!
    #find a given number of Fourier coefficients from a cubic-spline interpolated probability distribution function
    theta_sample_points = np.linspace(0,2*np.pi,2*terms,endpoint=False)
    import scipy.fft 
    c = scipy.fft.rfft(csi_pdf(theta_sample_points))/np.size(theta_sample_points)
    terms = np.size(c)-1 #because the a_0 term is included in c

    a_0  = np.real(c[0])
    a_n = 2*np.real(c[-terms:])
    b_n = -2*np.imag(c[-terms:])

    n = np.array((np.arange(1,terms+1)))
    def f(x):
        return a_0 + np.sum(a_n[:,None]*np.cos(n[:,None]*x)+b_n[:,None]*np.sin((n[:,None]*x)),axis=0)
        
    from scipy import integrate
    #normalise. so the area underneath = 1 (it's a CPDF)
    normalisation_factor = integrate.quad(f,0,2*np.pi)[0]
    a_0 = (a0*a_0)/normalisation_factor
    a_n = (a0*a_n)/normalisation_factor
    b_n = (a0*b_n)/normalisation_factor
    return a_0,a_n,b_n

def fourier_from_discrete_pdf(dpdf):   
    #find the number of Fourier coefficients from a discrete probability distribution function. Note: the number of terms in the fourier series is fixed by the length of the discrete pdf
    import scipy.fft 
    c = scipy.fft.rfft(dpdf)/np.size(dpdf)
    terms = np.size(c)-1 #because the a_0 term is included in c

    a_0 = np.real(c[0])
    a_n = 2*np.real(c[-terms:])
    b_n =-2*np.imag(c[-terms:])

    n = np.array((np.arange(1,terms+1)))
    def f(x):
        return a_0 + np.sum(a_n[:,None]*np.cos(n[:,None]*x)+b_n[:,None]*np.sin((n[:,None]*x)),axis=0)

    from scipy import integrate
    #normalise. so the area underneath = 1 (it's a CPDF)
    area = (integrate.quad(f,0,2*np.pi)[0])
    print("norm:{}".format(area))
    a_0 = a_0*((a_0*2*np.pi)/area)
    a_n = a_n*((a_0*2*np.pi)/area)
    b_n = b_n*((a_0*2*np.pi)/area)
    return a_0,a_n,b_n

def csi_pdf_from_data(wr_product,a_0=1):
    bin_centres = np.linspace(0,2*np.pi,np.size(wr_product)+1,endpoint=True)
    wr_product = np.append(wr_product,wr_product[0])
    inter_csi = CubicSpline(bin_centres,wr_product,bc_type='periodic')
    #normalise,so the area underneath is 1!
    from scipy import integrate
    norm = integrate.quad(inter_csi,0,2*np.pi)[0]
    csi_pdf =  CubicSpline(bin_centres,a_0*(wr_product/norm),bc_type='periodic')
    return csi_pdf

def discrete_pdf_from_csi_pdf(csi_pdf,no_bins,a_0=1):
    theta_sample_points = np.linspace(0,2*np.pi,no_bins,endpoint=False)
    #normalise, so the SUM(!) is 1!
    dpdf = a_0*(csi_pdf(theta_sample_points)/np.sum(csi_pdf(theta_sample_points)))
    return dpdf

def fourier_from_joint(frequency,avMagnitude):   
    # find the Fourier series representation of the (discrete!) joint frequency/avWindSpeed distribution in continous form.
    # Note: the number of terms in the fourier series is fixed by the length of the discrete pdf

    fa_0 = np.mean(frequency,dtype='float64')
    #conversion to a cpdf requires normalisation of the frequency area to 1
    joint_distribution = frequency*avMagnitude*(1/(2*np.pi*fa_0))
    
    import scipy.fft
    c = scipy.fft.rfft(joint_distribution)/np.size(joint_distribution)
    terms = np.size(c)-1 #because the a_0 term is included in c
    a_0 = np.real(c[0])
    a_n = 2*np.real(c[-terms:])
    b_n =-2*np.imag(c[-terms:])

    return a_0,a_n,b_n

def fourier_from_joint_savgol(frequency,avMagnitude,width=11,order=3):   
    #DEPRECIATED
    # find the Fourier series representation of the (discrete!) joint frequency/avWindSpeed distribution in continous form.
    # Note: the number of terms in the fourier series is fixed by the length of the discrete pdf

    fa_0 = np.mean(frequency,dtype='float64')
    #conversion to a cpdf requires normalisation of the frequency area to 1
    joint_distribution = frequency*avMagnitude*(1/(2*np.pi*fa_0))

    from scipy.signal import savgol_filter
    #(artificially) smooth the data (i know, this should be done by fitting a distribution / FVM / kernel regression, (and you could argue smoothing gives an advantage to the low-pass Fourier representation) but that is another study, and it is reasonable to assume the data is smooth)
    joint_distribution = savgol_filter(joint_distribution,width,order) 

    import scipy.fft
    c = scipy.fft.rfft(joint_distribution)/np.size(joint_distribution)
    terms = np.size(c)-1 #because the a_0 term is included in c
    a_0 = np.real(c[0])
    a_n = 2*np.real(c[-terms:])
    b_n =-2*np.imag(c[-terms:])

    return a_0,a_n,b_n

def cjd_from_binned_data(frequency,avMagnitude,width=11,order=3):   
    # find the Fourier series representation of the (discrete!) joint frequency/avWindSpeed distribution in continous form.
    # Note: the number of terms in the fourier series is fixed by the length of the discrete pdf
   
    from scipy.signal import savgol_filter
    #(artificially) smooth the data (i know, this should be done by fitting a distribution / FVM / kernel regression, (and you could argue smoothing gives an advantage to the low-pass Fourier representation) but that is another study, and it is reasonable to assume the data is smooth)
    djd = frequency*avMagnitude
    djd = savgol_filter(djd,width,order) 
    fa_0 = np.mean(frequency,dtype='float64')
    #conversion to a cpdf requires normalisation of the frequency area to 1
    norm_djd = djd*(1/(2*np.pi*fa_0))

    import scipy.fft
    c = scipy.fft.rfft(norm_djd)/np.size(norm_djd)
    terms = np.size(c)-1 #because the a_0 term is included in c
    a_0 = np.real(c[0])
    a_n = 2*np.real(c[-terms:])
    b_n =-2*np.imag(c[-terms:])
    fourier_coeffs = a_0,a_n,b_n
    
    #define a continous function for the exact integration
    n = np.array((np.arange(1,a_n.size+1))) #halve open things
    def cjd_func(x):
        return a_0 + np.sum(a_n[:,None]*np.cos(n[:,None]*x)+b_n[:,None]*np.sin((n[:,None]*x)),axis=0)

    return fourier_coeffs,cjd_func,a_0*2*np.pi

def djd_from_cjd_func(cjd_func,a_0,no_bins):
    sample_points = np.linspace(0,2*np.pi,no_bins,endpoint=False)
    djd = cjd_func(sample_points)*(a_0*2*np.pi)/(np.sum(cjd_func(sample_points)))
    return djd

def truncate_fourier(fourier_coeffs,terms):
    a_0,a_n,b_n = fourier_coeffs
    if terms > a_n.size:
        raise ValueError("The number of Fourier terms requested is greater than the length of the Fourier series")
    a_n = a_n[0:terms]
    b_n = b_n[0:terms]
    return a_0,a_n,b_n

def get_example_jd_func(custom=None,site=1,draw=False,no_bins=100):
    
    if custom==1:
        print("uniform wind rose selected")
        def cjd_func(theta):
            return (1/(2*np.pi))*np.ones(np.size(theta))
        djd = np.ones(no_bins)/no_bins
                
    else:
        print("site {} selected".format(site))
        filepath = r"C:\Users\Work\OneDrive - Durham University\4th Year Engineering\ENGI4093 Final Year Project\Python and Writing\WindRoseData\site"+str(site)+".csv"
        data = np.loadtxt(open(filepath,"rb"), delimiter=",", skiprows=2)

        bin_centres,frequency,avmagnitude,magMean = binWindData(data[:,5],data[:,6],bin_width=5)

        fourier_coeffs,cjd_func,U_av,djd = cjd_from_binned_data(frequency,avmagnitude)

    if draw==True:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5,5),dpi=400)
        sax = fig.add_subplot(projection='polar')
        xt = np.linspace(0,2*np.pi,200)
        sax.set_theta_direction(-1)
        sax.set_theta_zero_location('N')
        sax.plot(xt,cjd_func(xt))

    return cjd_func,djd