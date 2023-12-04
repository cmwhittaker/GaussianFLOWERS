#%% These are all the wake calculating functions
import numpy as np
from pathlib import Path
from floris.tools import FlorisInterface
from .helpers import adaptive_timeit,deltaU_by_Uinf_f,find_relative_coords
from scipy.interpolate import NearestNDInterpolator
from .flowers_interface import FlowersInterface

def floris_FULL_timed_aep(fl_wr, thetaD_i, layout, turb, wake=True, timed=True):
    """ 
    calculates the aep of a wind farm subject to directions theta_i Settings are taken from the "floris_settings.yaml". 
    The execution time is measured by the adaptive_timeit function (5 repeats over ~1.5 seconds) - this shouldn't effect the aep result.

    Args:
        fl_wr: floris wind rose object
        theta_i (bins,): In radians! Angle of bin (North is 0, clockwise +ve -to match compass bearings)
        layout (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines. Normalised by rotor diameter!
        turb (turbine obj) : turbine object, must have turb.D attribute to unnormalise layout
        wake (boolean): set False to run fi.calculate_no_wake()
        timed (boolean) : set False to run without timings (timing takes 4-8sec)

    Returns:
        pow_j (nt,) : aep of induvidual turbines
        time : lowest execution time measured by adaptive timeit 
    """    
    settings_path = Path("utilities") / "floris_settings.yaml"
    fi = FlorisInterface(settings_path)

    wd_array = np.array(fl_wr.df["wd"].unique(), dtype=float)
    if len(thetaD_i) != len(wd_array):
        raise ValueError("Floris is using a different amount of bins to FLOWERS (?!): len(thetaD_i) != len(wd_array)")
    ws_array = np.array(fl_wr.df["ws"].unique(), dtype=float)
    wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
    
    freq_interp = NearestNDInterpolator(fl_wr.df[["wd", "ws"]],fl_wr.df["freq_val"])
    freq = freq_interp(wd_grid, ws_grid)
    freq_2D = freq / np.sum(freq)
    
    turb_type = [turb.name,]
    fi.reinitialize(layout_x=turb.D*layout[:,0], layout_y=turb.D*layout[:,1],wind_directions=wd_array,wind_speeds=ws_array,time_series=False,turbine_type=turb_type)

    if wake:
        _,time = adaptive_timeit(fi.calculate_wake,timed=timed)
    else:
        _,time = adaptive_timeit(fi.calculate_no_wake,timed=timed)

    pow_j = np.nansum(fi.get_turbine_powers()*freq_2D[...,None],axis=(0,1)) 

    return pow_j/(1*10**6), time

def floris_AV_timed_aep(U_i, P_i, thetaD_i, layout, turb, wake=True, timed=True):
    """ 
    calculates the aep of a wind farm subject to directions theta_i with average bin velocity U_i and probability P_i. Settings are taken from the "floris_settings.yaml". 
    The execution time is measured by the adaptive_timeit function (5 repeats over ~1.5 seconds) - this shouldn't effect the aep result.

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

    settings_path = Path("utilities") / "floris_settings.yaml"
    fi = FlorisInterface(settings_path)
    fi.reinitialize(wind_directions=thetaD_i, wind_speeds=U_i, time_series=True, layout_x=turb.D*layout[:,0], layout_y=turb.D*layout[:,1])

    if wake:
        _,time = adaptive_timeit(fi.calculate_wake,timed=timed)
    else:
        _,time = adaptive_timeit(fi.calculate_no_wake,timed=timed)

    aep_array = fi.get_turbine_powers()
    pow_j = np.sum(P_i[:, None, None]*aep_array, axis=0)  # weight average using probability
    return pow_j/(1*10**6), time

def jflowers(Fourier_coeffs1,
             layout1,layout2,
             turb,
             K,
             c_0,
             RHO=1.225):
    
    """ 
    aep of wind farm using outlined in WES LoCascio 2022
    adapted from the code developed by Michae LoCascio,
    available at https://github.com/locascio-m/flowers/blob/main/flowers_interface.p

    Args:
        Fourier_coeffs1: tuple of Fourier coefficients (a_0, a_n, b_n) found on the series (1 - np.sqrt(1 - turb.Ct_f(U_i))) * U_i*P_i*len(P_i1)/(2*np.pi). U_i, P_i are the average bin speed and bin frequency respectively. The information would be reconstructed with a_0/2 + (a_n*np.cos(n*theta_b)+b_n*np.sin(n*theta_b). 
        layout (nt,2): coordinates ((x1,y1),(x2,y2) ... (xt_nt,yt_nt)) etc. of turbines. Normalised by rotor DIAMETER!
        turb (turbine obj) : turbine object, has turb.A (turbine swept area) attribute and turb.Cp_f() (turbine power coefficient) method.
        K (1,) : JENSEN wake growth parameter (different from the Gaussian )
        wake (boolean): set False to run fi.calculate_no_wake()
        timed (boolean) : set False to run without timings (timing takes 4-8sec)

    Returns:
        pow_j (nt,) : aep of induvidual turbines
        time : lowest execution time measured by adaptive timeit 
    """   

    #"my" implementation of the Jesen FLOWERS method
    # as outlined in WES LoCascio 2022
    
    R,THETA = find_relative_coords(layout1,layout2)

    a_0, a_n, b_n = Fourier_coeffs1 #unpack
    
    # Set up mask for rotor swept area
    mask_area = np.where(R<=0.5,1,0) #true within radius

    # Critical polar angle of wake edge (as a function of distance from turbine)
    theta_c = np.arctan(
        (1 / (2*R) + K * np.sqrt(1 + K**2 - (2*R)**(-2)))
        / (-K / (2*R) + np.sqrt(1 + K**2 - (2*R)**(-2)))
        ) 
    theta_c = np.nan_to_num(theta_c)
    
    # Contribution from zero-frequency Fourier mode
    du = a_0 * theta_c / (2 * K * R + 1)**2 * (
        1 + (2*(theta_c)**2 * K * R) / (3 * (2 * K * R + 1)))
    
    # Reshape variables for vectorized calculations
    m = np.arange(1, len(a_n)+1) #half open interval
    a = a_n[None, None,:] 
    b = b_n[None, None,:] 
    R = R[:, :, None]
    THETA = THETA[:, :, None] 
    theta_c = theta_c[:, :, None] 

    # Vectorized contribution of higher Fourier modes
    du += np.sum(
        (2*(a * np.cos(m*THETA) + b * np.sin(m*THETA)) / (m * (2 * K * R + 1))**3 *
        (
        np.sin(m*theta_c)*(m**2*(2*K*R*(theta_c**2+1)+1)-4*K*R)+ 4*K*R*theta_c*m *np.cos(theta_c * m))
        ), axis=2)
    # Apply mask for points within rotor radius
    du = np.where(mask_area,a_0,du)
    np.fill_diagonal(du, 0.) #stop self-produced wakes (?)
    # Sum power for each turbine
    du = np.sum(du, axis=1) #superposistion sum
    wav = (c_0*np.pi - du)
    alpha = turb.Cp_f(wav)*wav**3 
    aep = (0.5*turb.A*RHO*alpha)/(1*10**6)

    return aep

def LC_flowers_timed_aep(U_i,P_i,thetaD_i,layout, turb, K=0.05,Nterms =36,timed=True):
    #theta_i in DEGREES!     
    flower_int = FlowersInterface(U_i,P_i,thetaD_i, layout, turb,num_terms=Nterms+1, k=K) 
    aep_func = lambda: flower_int.calculate_aep()
    #time execution
    pow_j,time = adaptive_timeit(aep_func,timed=timed)
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
    (the overlap in functionality between this and ntag, vect_num_Fs and caag is useful for validation)
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
                        The constant is passed to the function with the wav_Cp "weight-averaged" kwarg
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
    #(wind bins, turbines in farm, turbines in superposistion) e.g.
    #(72,5,5) for 72 binned 5 turbine farm 
    #(it should be (72,5,5-1) but I make sure lim>0 to stop turbines producing their own wake - I found the implementation was more intuitive than removing a column/row in the matrix)

    if np.any(np.abs(theta_i) > 10): #this is needed ... 
        raise ValueError("Did you give num_F degrees?")
        
    def get_sort_index(layout,theta_i):
        #sorts turbines from furthest upwind in wind orientated frame           
        def rotate_layout(layout,rot):
            #rotates layout anticlockwise by angle rot 
            Xt,Yt = layout[:,0],layout[:,1]
            rot_Xt = Xt * np.cos(rot) - Yt * np.sin(rot)
            rot_Yt = Xt * np.sin(rot) + Yt * np.cos(rot) 
            layout_r = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
            return layout_r
        #from wind orientated frame, rotation is opposite
        layout_r = rotate_layout(layout,-theta_i)

        sort_index = np.argsort(layout_r[:,0]) #sort index, with furthest upwind (x_n < x_{n+1}) first
        return sort_index
    
    def soat(a): #Sum over Axis Two (superposistion axis sum)
        return np.sum(a,axis=2)

    x_n,y_n = layout[:,0],layout[:,1]
    X,Y = plot_points[:,0],plot_points[:,1] #flatten arrays

    DUt_ijk = np.zeros((len(U_i),len(x_n),len(x_n)))
    Uwt_ij = U_i[:,None]*np.ones((1,x_n.shape[0])) #turbine Uws

    DUff_ijk = np.zeros((len(U_i),len(X),len(x_n)))
    Uwff_ij = U_i[:,None]*np.ones(((1,X.shape[0])))
    
    #the actual calculation loop
    for i in range(len(U_i)): #for each wind direction
        
        sort_index = get_sort_index(layout,theta_i[i]) #find
        layout_s = layout[sort_index] #and reorder based on furthest upwind
        
        for m in range(len(x_n)): #for each turbine in superposistion
            x_m, y_m = layout_s[m,0],layout_s[m,1]       
            #calculate relative turbine locations
            Rt = np.sqrt((x_n-x_m)**2+(y_n-y_m)**2)
            THETAt = np.arctan2(y_n-y_m,x_n-x_m) - theta_i[i] 
            #calculate relative plot_points (flow field ff) locations
            Rff = np.sqrt((X-x_m)**2+(Y-y_m)**2)
            THETAff = np.arctan2(Y-y_m,X-x_m) - theta_i[i]
            
            #then there are a few ways of defining the thrust coefficient
            if Ct_op == 1: #base on local velocity
                Ct = turb.Ct_f(Uwt_ij[i,m])
            elif Ct_op == 2: #base on global inflow
                Ct = turb.Ct_f(U_i[i]) 
            elif Ct_op == 3: #based on some constant
                if wav_Ct == None:
                    raise ValueError("For option 3 provide wav_Ct")
                Ct = wav_Ct
            else:
                raise ValueError("Ct_op is not supported")

            DUt_ijk[i,:,m] = deltaU_by_Uinf_f(Rt,THETAt,Ct,K,u_lim,ex)
            Uwt_ij[i,:] = Uwt_ij[i,:] - U_i[i]*DUt_ijk[i,:,m] #sum over superposistion for each turbine (turbine U_ws)            
            DUff_ijk[i,:,m] = deltaU_by_Uinf_f(Rff,THETAff,Ct,K,u_lim,ex)
            
            Uwff_ij[i,:] = Uwff_ij[i,:] - U_i[i]*DUff_ijk[i,:,m] #sum over superposistion for eah turbine (flow field)
    
    num_Fs.DUt_ijk = DUt_ijk #debugging
    num_Fs.DUff_ijk = DUff_ijk #(slightly hacky) this is for the cross-term plot (i don't want to change the signature just for this one use case)
    
    if cross_ts: #INcluding cross terms
        if cube_term == False:
            raise ValueError("Did you mean to neglect the cross terms?")
        Uwt_ij_cube = Uwt_ij**3 #simply cube the turbine velocities
    
    else: #EXcluding cross terms (soat = Sum over Axis Two (third axis!)
        Uwt_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(DUt_ijk) + 3*soat(DUt_ijk**2) - cube_term*soat(DUt_ijk**3)) #optionally neglect the cubic term with the cube_term option

    #then there are a few ways of finding the power coefficient / calculating power
    if Cp_op == 1: # base on local wake velocity C_p(U_w)
        Cp_ij = turb.Cp_f(Uwt_ij)
        pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uwt_ij_cube),axis=0)/(1*10**6)
    elif Cp_op == 2: #base on global inflow C_p(U_\infty) 
        Cp_ij = turb.Cp_f(U_i)[:,None]
        pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uwt_ij_cube),axis=0)/(1*10**6)
    elif Cp_op == 3: #based on some constant (weight-averaged)
        if wav_Cp is None:
            raise ValueError("For Cp option 3 provide wav_Cp")
        pow_j = 0.5*turb.A*RHO*wav_Cp*np.sum(P_i[:,None]*(Uwt_ij**3),axis=0)/(1*10**6)
    elif Cp_op == 4: #Gaussian equivlent of Jensen FLOWERS method (as found in the final year project)
        if Ct_op != 3:
            raise ValueError("This has no analyical equivalent Ct_op should probably be 3")
        alpha = np.sum(P_i[:,None]*Uwt_ij,axis=0) #the weight average velocity field
        pow_j = 0.5*turb.A*RHO*turb.Cp_f(alpha)*alpha**3/(1*10**6)
    elif Cp_op == 5: 
        #This is Cp^1/3 method (may be incorrect) from FLOWERS v2.0
        if Ct_op != 3:
            raise ValueError("Cp_op should be 3")
        alpha = np.sum((turb.Cp_f(Uwt_ij))**(1/3)*P_i[:,None]*Uwt_ij,axis=0) #the weight average velocity field
        pow_j = 0.5*turb.A*RHO*alpha**3/(1*10**6)
    else:
        raise ValueError("Cp_op value is not supported")
    #(j in Uwff_j here is indexing the meshgrid)
    Uwt_j = np.sum(P_i[:,None]*Uwt_ij,axis=0)
    Uwff_j = np.sum(P_i[:,None]*Uwff_ij,axis=0) #probability weighted flow field
    
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
    if np.any(np.abs(theta_i) > 10): #this is needed ... 
        raise ValueError("Did you give me degrees?")
    
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
    Uwt_ij = U_i[:,None]*(1-np.sum(deltaU_by_Uinf_f(r_ijk,theta_ijk,ct_ijk,K,u_lim,ex),axis=2)) #wake velocity at turbine locations
    pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(turb.Cp_f(Uwt_ij)*Uwt_ij**3),axis=0)/(1*10**6)
    return pow_j,Uwt_ij

def ntag_PA(Fourier_coeffs3_PA,
            layout1,layout2,
            turb,
            K,
            wav_Ct,
            u_lim=3,
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
        alpha (nt,2) | (plot_points,2) : "energy content" (Cp(U)*P*U**3) of the wind at turbine locations or plot_points
    """
    r_jk,theta_jk = find_relative_coords(layout1,layout2)  #find relative posistions
    theta_jk = np.mod(theta_jk + np.pi, 2 * np.pi) - np.pi #fix domain

    A_n,Phi_n = Fourier_coeffs3_PA
    a_0 = 2*A_n[0] #because A_n[0] = a_0 / 2

    EP = 0.2*np.sqrt((1+np.sqrt(1-wav_Ct))/(2*np.sqrt(1-wav_Ct))) #initial wake expansion parameter

    #auxilaries 
    n = np.arange(0,A_n.size,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(wav_Ct/8)-EP)/K
    lim = np.where(lim<u_lim,u_lim,lim) #pick greater from u_lim and lim
    if np.any((r_jk<lim) & (r_jk != 0)):
        raise ValueError("turbines within the invalid region, this will likely cause erroneously low AEP")
    #if turbine 1 is posistioned adjacent to turbine 2, neither upwind or downwind (""inline with each other, perpendicular to the wind direction"")", if within the r limit, turbine 1 will be waked by turbine 2 - which is not realistic (or atleast not as described by Bastankah 2014)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(wav_Ct/(8*(K*r_jk+EP)**2))))) 
    
    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    A_n = A_n[None,None,:]
    Phi_n = Phi_n[None,None,:]
    theta_b = theta_jk[:,:,None]
    #more auxilaries
    fs = A_n*np.cos(n_b*theta_b+Phi_n) #fourier series (including DC!)

    def term(a):
        cnst_term = ((np.sqrt(2*np.pi*a)*sigma)/(a))*(sqrt_term**a)
        mfs = (np.sum(np.exp(-((sigma_b*n_b)**2)/(2*a))*(fs),axis=-1)) #modified Fourier series
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

    A_n,Phi_n = Fourier_coeffs_noCp_PA
    a_0 = 2*A_n[0] #because A_n[0] = a_0 / 2

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
