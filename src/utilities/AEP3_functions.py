#%% These are all the wake calculating functions

def floris_timed_aep(U_i, P_i, theta_i, layout, turb, wake=True, timed=True):
    """ 
    calculates the aep of a wind farm subject to directions theta_i with average bin velocity U_i and probability P_i. Settings are taken from the "floris_settings.yaml". The execution time is measured by the adaptive_timeit function (5 repeats over ~1.5 seconds) - this shouldn't effect the aep result.
    

    Args:
        U_i (bins,): Average wind speed of bin
        P_i (bins,): Probability of bin
        theta_i (bins,): In radians! Angle of bin (North is 0, clockwise +ve -to match compass bearings)
        layout (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines. Normalised by rotor diameter!
        turb (turbine obj) : turbine object, must have turb.D attribute to unnormalise layout
        wake (boolean): set False to run fi.calculate_no_wake()
        timed (boolean) : set False to run without timings (timing takes 4-8sec)

    Returns:
        pow_j (nt,) : aep of induvidual turbines
        time : lowest execution time measured by adaptive timeit 
    """
    from pathlib import Path
    from floris.tools import FlorisInterface
    from utilities.helpers import adaptive_timeit
    settings_path = Path("utilities") / "floris_settings.yaml"
    fi = FlorisInterface(settings_path)
    fi.reinitialize(wind_directions=theta_i, wind_speeds=U_i, time_series=True, layout_x=turb.D*layout[:,0], layout_y=turb.D*layout[:,1])

    if wake:
        _,time = adaptive_timeit(fi.calculate_wake,timed=timed)
    else:
        _,time = adaptive_timeit(fi.calculate_no_wake,timed=timed)

    aep_array = fi.get_turbine_powers()
    pow_j = np.sum(P_i[:, None, None]*aep_array, axis=0)  # weight average using probability
    return pow_j/(1*10**6), time

import numpy as np
def num_Fs(U_i,P_i,theta_i,
           layout,plot_points, #this is the comp domain
           turb,
           K,
           RHO=1.225,
           u_lim=None,
           Ct_op=1,wav_Ct=None,
           Cp_op=1,wav_Cp=None,
           cross_ts=True,ex=True,cube_term=True):
    """ 
    "general purpose" numerical equivalent of the many different options to handle the aep calcuation integration. 

    Args:
        U_i (bins,): Average wind speed of bin
        P_i (bins,): Probability of bin
        theta_i (bins,): In radians! Angle of bin (North is 0, clockwise +ve -to match compass bearings)
        layout (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines
        plot_points (n_grid_points,2): the contouf meshgrid converted to coordinates e.g. ((xx1,yy1),(xx2,yy2) ...) (for the nice flow field plots)
        turb (turbine object): must have: turb.A area attribute, Ct_f and Cp_f methods for interpolating the thrust and power cofficient curve
        K (float): Gaussian wake expansion parameter
        RHO (float): assumed atmospheric density
        u_lim (float): user defined invalid radius limit. This sets a radius around the turbine where the deficit is zero (useful for plotting)

        now onto the options:
        Ct_op (int): This selects a way to find the thrust coefficient
                     Ct_op == 1: Local thrust coefficient Ct(U_w)
                        (this requires the thrust coefficients to be found in turn - as far as I'm aware you cannot vectorise this)
                     Ct_op == 2: Global thrust coefficient Ct(U_\infty)
                        (the thrust coefficient is based on the leading turbine, this CAN be vectorised, but for simplicity I wrote another function "vect_num_F" that does this)
                     Ct_op == 3: Constant thrust coefficient
                        The constant is passed to the function with the wav_Ct "weight-averaged" kwarg

        Cp_op (int): This selects a way to find the power coefficient / a way to calculate the power
                     Ct_op == 1: Local power coefficient Ct(U_w)
                     Ct_op == 2: Global power coefficient Ct(U_\infty)
                     Ct_op == 3: Constant power coefficient
                        (this isn't really used...)
                        The constant is passed to the function with the wav_Ct "weight-averaged" kwarg
                     Ct_op == 4: This turns num_Fs into the numerical equivalent of the "Jensen FLOWERS" approach but with a Guassian wake model.
                     Ct_op == 5: This turns num_Fs into the numerical equivalent of the "Jensen FLOWERS v2.0" (may be inccorect) approach but with a Guassian wake model.

        Options 4 and 5 are meant to mimick analytical methods, so there is some limited functionality to stop the user selecting combinations that have no analytical equivalent

        cross_ts (bool): True or False to include or neglect the cross terms
        ex (bool): "EXact wake model" - False to use small angle approximation
        cube_term: "Include the cubic term in the binomial expasion?" this was suggested by Majid and does increase accuracy

    Returns:
        pow_j (nt,) : aep of induvidual turbines
        Uwt_j (nt,2) : wake velocity at turbine locations
        Uwff_j (n_grid_points,2) : wake velocity at flow field points (for plotting)
    """
    #generally, the numpy arrays are indexed as:
    #(i,j,k)
    #(wind bins, turbines in farm, turbines in superposistion)

    if np.any(np.abs(theta_i) > 10): #this is needed ...
        raise ValueError("Did you give num_F degrees?")
    
    def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct))) #initial expansion width: (eq.6 + 19 +21 in Bastankah 2014 - don't forget eq21!)
        if u_lim != None: #override the limit with the user defined radius
            lim = u_lim
        else:
            lim = (np.sqrt(Ct/8)-ep)/K #invalid region
            lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) 
        
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
    
    def get_sort_index(layout,rot):
        #rotate layout coordinates by rot clockwise (in radians)
        Xt,Yt = layout[:,0],layout[:,1]
        rot_Xt = Xt * np.cos(rot) + Yt * np.sin(rot)
        rot_Yt = -Xt * np.sin(rot) + Yt * np.cos(rot) 
        layout = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
        sort_index = np.argsort(-layout[:, 1]) #sort index, with furthest upwind first
        return sort_index
    
    def soat(a): #Sum over Axis Two (superposistion axis sum)
        return np.sum(a,axis=2)

    Xt,Yt = layout[:,0],layout[:,1]
    X,Y = plot_points[:,0],plot_points[:,1] #flatten arrays

    DUt_ijk = np.zeros((len(U_i),len(Xt),len(Xt)))
    Uwt_ij = U_i[:,None]*np.ones((1,Xt.shape[0])) #turbine Uws

    DUff_ijk = np.zeros((len(U_i),len(X),len(Xt)))
    Uwff_ij = U_i[:,None]*np.ones(((1,X.shape[0])))
    flag = True
    #the actual calculation loop
    for i in range(len(U_i)): #for each wind direction
        
        sort_index = get_sort_index(layout,-theta_i[i]) #find
        layout = layout[sort_index] #and reorder based on furthest upwind
        
        for k in range(len(Xt)): #for each turbine in superposistion
            xt, yt = layout[k,0],layout[k,1]       
            #calculate relative turbine locations
            Rt = np.sqrt((Xt-xt)**2+(Yt-yt)**2)
            THETAt = np.pi/2 - np.arctan2(Yt-yt,Xt-xt) - theta_i[i]
            #calculate relative plot_points (flow field ff) locations
            Rff = np.sqrt((X-xt)**2+(Y-yt)**2)
            THETAff = np.pi/2 - np.arctan2(Y-yt,X-xt) - theta_i[i]
            
            #then there are a few ways of defining the thrust coefficient
            if Ct_op == 1: #base on local velocity
                Ct = turb.Ct_f(Uwt_ij[i,k])
            elif Ct_op == 2: #base on global inflow
                Ct = turb.Ct_f(U_i[i]) 
            elif Ct_op == 3:
                if wav_Ct == None:
                    raise ValueError("For option 3 provide WAV_CT")
                Ct = wav_Ct
                if flag:
                    print(f"using WAV_CT: {wav_Ct:.2f}")
                    flag = False
            else:
                raise ValueError("Ct_op is not supported")

            DUt_ijk[i,:,k] = deltaU_by_Uinf_f(Rt,THETAt,Ct,K)
            Uwt_ij[i,:] = Uwt_ij[i,:] - U_i[i]*DUt_ijk[i,:,k] #sum over superposistion for each turbine (turbine U_ws)
            
            DUff_ijk[i,:,k] = deltaU_by_Uinf_f(Rff,THETAff,Ct,K)
            Uwff_ij[i,:] = Uwff_ij[i,:] - U_i[i]*DUff_ijk[i,:,k] #sum over superposistion for eah turbine (flow field)
    
    num_Fs.DUff_ijk = DUff_ijk #(slightly hacky) this is for the cross-term plot (i don't want to change the signature just for this one use case)
    
    if cross_ts: #INcluding cross terms
        if cube_term == False:
            raise ValueError("Did you mean to neglect the cross terms?")
        Uwt_ij_cube = Uwt_ij**3 #simply cube the turbine velocities
    else: #EXcluding cross terms (soat = Sum over Axis Two (third axis!)
        Uwt_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(DUt_ijk) + 3*soat(DUt_ijk**2) - cube_term*soat(DUt_ijk**3)) #optionally neglect the cubic term with the cube_term option

    #then there are a few ways of finding the power coefficient / calculating power
    if Cp_op == 1: #power coeff based on local wake velocity
        Cp_ij = turb.Cp_f(Uwt_ij)
        pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uwt_ij_cube),axis=0)/(1*10**6)
    elif Cp_op == 2: #power coeff based on global inflow U_infty
        Cp_ij = turb.Cp_f(U_i)[:,None]
        pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uwt_ij_cube),axis=0)/(1*10**6)
    elif Cp_op == 3: #use weight averaged Cp
        if wav_Cp is None:
            raise ValueError("For Cp option 3 provide wav_Cp")
        pow_j = 0.5*turb.A*RHO*wav_Cp*np.sum(P_i[:,None]*(Uwt_ij**3),axis=0)/(1*10**6)
    elif Cp_op == 4: #the old way (found analytical version in FYP)
        if Ct_op != 3:
            raise ValueError("This has no analyical equivalent Ct_op should probably be 3")
        alpha = np.sum(P_i[:,None]*Uwt_ij,axis=0) #the weight average velocity field
        pow_j = 0.5*turb.A*RHO*turb.Cp_f(alpha)*alpha**3/(1*10**6)
    elif Cp_op == 5: 
        #This is Cp^1/3 method(may be incorrect)
        if Ct_op != 3:
            raise ValueError("Cp_op should be 3")
        alpha = np.sum((turb.Cp_f(Uwt_ij))**(1/3)*P_i[:,None]*Uwt_ij,axis=0) #the weight average velocity field
        pow_j = 0.5*turb.A*RHO*alpha**3/(1*10**6)
    else:
        raise ValueError("Cp_op value is not supported")
    #(j in Uwff_j here is indexing the meshgrid)
    Uwt_j = np.sum(P_i[:,None]*Uwt_ij,axis=0)
    Uwff_j = np.sum(P_i[:,None]*Uwff_ij,axis=0) #weighted flow field
    
    return pow_j,Uwt_j,Uwff_j 

def vect_num_F(U_i,P_i,theta_i,
               layout1,layout2, 
               turb,
               K,
               RHO=1.225,
               u_lim=None,
               Ct_op=2,wav_Ct=None,
               Cp_op=1,  
               ex=True):

    """ 
    A vectorised/ optimised version of num_F_v02. 
    This means it can be compared in terms of performance AND accuracy to the "Gaussian FLOWERS" method.
    Vectorisation removes the choice of find Ct based on local wake velocity. It must either be global or constant.
    Only a power cofficient based on local wake velocity is supported (this is the best option, so there is no reason to use a worse choice)

    Args:
        U_i (bins,): Average wind speed of bin
        P_i (bins,): Probability of bin
        theta_i (bins,): In radians! Angle of bin (North is 0, clockwise +ve -to match compass bearings)
        layout1 (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines
        layout2 (nt,2) OR (n_grid_points,2): when layout2 = layout1, Uwt_j gives the power/wake velocity at the turbine locations. if layout2 = plot_points, (the power result is meaningless) Uwt_j is the wake velocity at the plot points (useful for plotting)
        turb (turbine object): must have: turb.A area attribute, Ct_f and Cp_f methods for interpolating the thrust and power cofficient curve
        K (float): Gaussian wake expansion parameter
        RHO (float): assumed atmospheric density
        u_lim (float): user defined invalid radius limit. This sets a radius around the turbine where the deficit is zero (useful for plotting)

        now onto the options:
        Ct_op (int): This selects a way to find the thrust coefficient
                     Ct_op == 2: Global thrust coefficient Ct(U_\infty)
                        (the thrust coefficient is based on the leading turbine)
                     Ct_op == 3: Constant thrust coefficient
                        The constant is passed to the function with the wav_Ct "weight-averaged" kwarg
        Cp_op (int): This selects a way to find the power coefficient / a way to calculate the power
                     Ct_op == 1: Local power coefficient Ct(U_w)
        ex (bool): "EXact wake model" - False to use small angle approximation

    Returns:
        pow_j (nt,) or (n_grid_points,) : aep of induvidual turbines (or meaningless: the aep if there was turbine at every plot point )
        Uwt_j (nt,) or (n_grid_points) : wake velocity at turbine locations (if layout2 is plot_points, this will give the wake velocity at the plot points, which is useful for plotting)
    """    
    def deltaU_by_Uinf_f(r,theta,Ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
        if u_lim != None:
            lim = u_lim
        else:
            lim = (np.sqrt(Ct/8)-ep)/K
            lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) 
        
        theta = theta + np.pi #the wake lies opposite!
        if ex: #use full 
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*np.cos(theta)+ep)**2))))*(np.exp(-(r*np.sin(theta))**2/(2*(K*r*np.cos(theta)+ep)**2)))
            deltaU_by_Uinf = np.where(r*np.cos(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit  
        else: #otherwise use small angle approximations
            theta = np.mod(theta-np.pi,2*np.pi)-np.pi
            U_delta_by_U_inf = (1-np.sqrt(1-(Ct/(8*(K*r*1+ep)**2))))*(np.exp(-(r*theta)**2/(2*(K*r*1+ep)**2)))          
            deltaU_by_Uinf = np.where(r>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
        
        return deltaU_by_Uinf   

    #I sometimes use this function to find the wake field for plotting, so find relative posistions to plot points not the layout 
    #when layout2 = plot_points it finds wake at the turbine posistions
    r_jk,theta_jk = find_relative_coords(layout1,layout2) #find theta relative to each turbine and each turbine in superposistion
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None] #find theta when wind direction varies
    r_ijk =  np.broadcast_to(r_jk[None,:,:],theta_ijk.shape) 
    if Ct_op == 1:
        raise ValueError("Local Ct is not supported by this function") #this has to be done in turn
    elif Ct_op == 2: #global Ct *(based on freestream inflow speed)
        ct_ijk = np.broadcast_to(turb.Ct_f(U_i)[...,None,None],r_ijk.shape)
    elif Ct_op == 3:
        if wav_Ct == None:
            raise ValueError("For option 3 provide WAV_CT")
        ct_ijk = np.broadcast_to(wav_Ct,r_ijk.shape)
    else:
        raise ValueError("No Ct option selected")
    #
    if Cp_op != 1:
        raise ValueError("Only option 1 is supported for Cp ")
    #power coefficient based on local inflow
    
    Uwt_ij = U_i[:,None]*(1-np.sum(deltaU_by_Uinf_f(r_ijk,theta_ijk,ct_ijk,K),axis=2)) #wake velocity at turbine locations
    pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(turb.Cp_f(Uwt_ij)*Uwt_ij**3),axis=0)/(1*10**6)
    return pow_j,Uwt_ij

from utilities.helpers import find_relative_coords
def ntag_PA(Fourier_coeffs3_PA,
            layout1,layout2,
            turb,
            K,
            wav_Ct,
            RHO=1.225):
    """ 
    "No cross Terms Analytical Gaussian (Phase Amplitude)" - The "Gaussian FLOWERS" method implemented 

    Args:
        Fourier_coeffs3_PA (list of np.arrays): tuple of Fourier coefficient (a_0,A_n,Phi_n), such that f(theta)= a_0/2 + (A_n*np.cos(n*theta+Phi_n)
        reconstructs the wind rose Cp(U(theta))*P(theta)*U(theta)**3
        layout1 (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines
        layout2 (nt,2) OR (n_grid_points,2): when layout2 = layout1, Uwt_j      gives the power/wake velocity at the turbine locations. if layout2 = plot_points, (the power result is meaningless) Uwt_j is the wake velocity at the plot points (useful for plotting)
        turb (turbine object): must have: turb.A area attribute, Ct_f and Cp_f methods for interpolating the thrust and power cofficient curve
        K (float): Gaussian wake expansion parameter
        wav_Ct (float): The constant thrust coefficient 
        RHO (float): assumed atmospheric density
        u_lim (float): user defined invalid radius limit. This sets a radius around the turbine where the deficit is zero (useful for plotting)

    Returns:
        pow_j (nt,) : aep of induvidual turbines
        alpha (nt,2) | (plot_points,2) : "energy content" (Cp*P*U**3) of the wind at turbine locations or plot_points
    """
    r_jk,theta_jk = find_relative_coords(layout1,layout2) #find relative posistions

    a_0,A_n,Phi_n = Fourier_coeffs3_PA

    EP = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))

    #auxilaries
    n = np.arange(1,A_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(wav_Ct/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(wav_Ct/(8*(K*r_jk+EP)**2)))))

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    A_n = A_n[None,None,:]
    Phi_n = Phi_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream
    #more auxilaries
    fs = A_n*np.cos(n_b*theta_b+Phi_n)
    nsigma = sigma_b*n_b

    def term(a):
        cnst_term = ((np.sqrt(2*np.pi*a)*sigma)/(a))*(sqrt_term**a)
        mfs = (a_0/2 + np.sum(np.exp(-((nsigma)**2)/(2*a))*(fs),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)

    #alpha is the 'energy' content of the wind
    alpha = (a_0/2)*2*np.pi - 3*term(1) + 3*term(2) #- term(3)
    #(I fully vectorised this and it ran slower ... so I'm sticking with this)
    #If it were vectorised using dimensions sparingly (e.g. don't broadcast everything to 4D (alpha,J,K,N) ) immediately) it might be faster
    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*turb.A*RHO*alpha)/(1*10**6)
    else: #farm wake visualisation, power is meaningless
        pow_j = np.nan
    return pow_j,alpha

def caag_PA(Fourier_coeffs_noCp_PA,
            layout1,layout2,
            turb,
            K,
            wav_Ct,
            RHO=1.225):
    
    """ 
    "Cubed of the Average Analytical Gaussian" - A Gaussian equivalent to the "Jensen FLOWERS" method

    Args:
        Fourier_coeffs_noCp_PA (list of np.arrays): tuple of Fourier coefficient (a_0,A_n,Phi_n), such that f(theta)= a_0/2 + (A_n*np.cos(n*theta+Phi_n)
        reconstructs the wind rose P(theta)*U(theta) (doesn't include Cp)
        layout1 (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines
        layout2 (nt,2) OR (n_grid_points,2): when layout2 = layout1, Uwt_j      gives the power/wake velocity at the turbine locations. if layout2 = plot_points, (the power result is meaningless) Uwt_j is the wake velocity at the plot points (useful for plotting)
        turb (turbine object): must have: turb.A area attribute, Ct_f and Cp_f methods for interpolating the thrust and power cofficient curve
        K (float): Gaussian wake expansion parameter
        wav_Ct (float): The constant thrust coefficient 
        RHO (float): assumed atmospheric density

    Returns:
        pow_j (nt,) : aep of induvidual turbines
        alpha (nt,2) | (plot_points,2) : "energy content" (P*U) of the wind at turbine locations or plot_points
    """

    r_jk,theta_jk = find_relative_coords(layout1,layout2)

    a_0,A_n,Phi_n = Fourier_coeffs_noCp_PA

    EP = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct)))

    #auxilaries
    n = np.arange(1,A_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(wav_Ct/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(wav_Ct/(8*(K*r_jk+EP)**2)))))
    cnst_term = sqrt_term*np.sqrt(2*np.pi)*sigma

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    A_n = A_n[None,None,:]
    Phi_n = Phi_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream
   
    cnst_term = ((np.sqrt(2*np.pi)*sigma))*sqrt_term
    mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2))*(A_n*np.cos(n_b*theta_b+Phi_n)),axis=-1)) #modified Fourier series 
        
    alpha = 2*np.pi*a_0/2 - np.sum(cnst_term*mfs,axis=-1)  #the weight average velocity at each turbine

    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*turb.A*RHO*turb.Cp_f(alpha)*alpha**3)/(1*10**6)
    else: #farm wake visualisation, power is meaningless
        pow_j = np.nan

    return pow_j,alpha
