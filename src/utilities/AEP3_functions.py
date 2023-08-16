#%% These are all the wake calculating functions 
import numpy as np

def num_Fs(U_i,P_i,theta_i,
           layout,plot_points, #this is the comp domain
           turb,
           K,
           RHO=1.225,
           u_lim=None,
           Ct_op=1,WAV_CT=None,
           Cp_op=1,WAV_CP=None,
           cross_ts=True,ex=True,cube_term=True):
      #this now finds the wakes of the turbines "in turn" so that it can support a thrust coefficient based on local inflow: Ct(U_w) not Ct(U_\infty) as previously.
    if np.any(np.abs(theta_i) > 10): #this is needed ...
        raise ValueError("Did you give num_F degrees?")
    
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
        
        return deltaU_by_Uinf  
    
    def get_sort_index(layout,rot):
        #rot:clockwise +ve
        Xt,Yt = layout[:,0],layout[:,1]
        rot_Xt = Xt * np.cos(rot) + Yt * np.sin(rot)
        rot_Yt = -Xt * np.sin(rot) + Yt * np.cos(rot) 
        layout = np.column_stack((rot_Xt.reshape(-1),rot_Yt.reshape(-1)))
        sort_index = np.argsort(-layout[:, 1]) #sort index, with furthest upwind first
        return sort_index
    
    def soat(a): #Sum over Axis Two (superposistion sum)
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
            #turbine locations 
            Rt = np.sqrt((Xt-xt)**2+(Yt-yt)**2)
            THETAt = np.pi/2 - np.arctan2(Yt-yt,Xt-xt) - theta_i[i]
            #these are the flow field mesh
            Rff = np.sqrt((X-xt)**2+(Y-yt)**2)
            THETAff = np.pi/2 - np.arctan2(Y-yt,X-xt) - theta_i[i]

            if Ct_op == 1: #base on local velocity
                Ct = turb.Ct_f(Uwt_ij[i,k])
            elif Ct_op == 2: #base on global inflow
                Ct = turb.Ct_f(U_i[i]) 
            elif Ct_op == 3:
                if WAV_CT == None:
                    raise ValueError("For option 3 provide WAV_CT")
                Ct = WAV_CT
                if flag:
                    print(f"using WAV_CT: {WAV_CT:.2f}")
                    flag = False
            else:
                raise ValueError("No Ct option selected")

            DUt_ijk[i,:,k] = deltaU_by_Uinf_f(Rt,THETAt,Ct,K)
            Uwt_ij[i,:] = Uwt_ij[i,:] - U_i[i]*DUt_ijk[i,:,k] #sum over k
            
            DUff_ijk[i,:,k] = deltaU_by_Uinf_f(Rff,THETAff,Ct,K)
            Uwff_ij[i,:] = Uwff_ij[i,:] - U_i[i]*DUff_ijk[i,:,k] #sum over k
    
    num_Fs.DUff_ijk = DUff_ijk #(slightly hacky) this is for the cross-term plot (i don't want to change the signature just for this one use case)
    #calculate power at the turbine location
    if cross_ts: #INcluding cross terms
        if cube_term == False:
            raise ValueError("Did you mean to neglect the cross terms?")
        Uwt_ij_cube = Uwt_ij**3
    else: #EXcluding cross terms (soat = Sum over Axis Two (third axis!)
        #cube_term neglects the cube term 
        Uwt_ij_cube = (U_i[:,None]**3)*(1 - 3*soat(DUt_ijk) + 3*soat(DUt_ijk**2) - cube_term*soat(DUt_ijk**3))

    if Cp_op == 1: #power coeff based on local wake velocity
        Cp_ij = turb.Cp_f(Uwt_ij)
        pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uwt_ij_cube),axis=0)/(1*10**6)
    elif Cp_op == 2: #power coeff based on global inflow U_infty
        Cp_ij = turb.Cp_f(U_i)[:,None]
        pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(Cp_ij*Uwt_ij_cube),axis=0)/(1*10**6)
    elif Cp_op == 3: #use weight averaged Cp
        if WAV_CP is None:
            raise ValueError("For Cp option 3 provide WAV_CP")
        pow_j = 0.5*turb.A*RHO*WAV_CP*np.sum(P_i[:,None]*(Uwt_ij**3),axis=0)/(1*10**6)
    elif Cp_op == 4: #the old way (found analytical version in FYP)
        alpha = np.sum(P_i[:,None]*Uwt_ij,axis=0) #the weight average velocity field
        pow_j = 0.5*turb.A*RHO*turb.Cp_f(alpha)*alpha**3/(1*10**6)
    elif Cp_op == 5: 
        #This is Cp^1/3 method(may be incorrect)
        if Ct_op != 3:
            raise ValueError("Cp_op should be 3")
        alpha = np.sum((turb.Cp_f(Uwt_ij))**(1/3)*P_i[:,None]*Uwt_ij,axis=0) #the weight average velocity field
        pow_j = 0.5*turb.A*RHO*alpha**3/(1*10**6)
    elif Cp_op == 6: 
        if Ct_op != 3 and cross_ts != False:
            raise ValueError("Ct_op should be 3 AND cross terms must be false")
        #This is the weird hybrid method
        alpha = np.sum(P_i[:,None]*Uwt_ij,axis=0) #the weight average velocity field
        pow_j = 0.5*turb.A*RHO*turb.Cp_f(alpha)*np.sum(P_i[:,None]*(Uwt_ij_cube),axis=0)/(1*10**6)
       
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
    # a vectorised/ optimised version of num_F_v02
    # this means it can be compared in terms of performance AND accuracy
    # vectorisation restricts choice of Ct_op to global or constant averaged (2 or 3)
    # there's no option to neglect the cross terms

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

    #I sometimes use this function to find the wake layout, so find relative posistions to plot points not the layout 
    #when plot_points = layout it finds wake at the turbine posistions
    r_jk,theta_jk = gen_local_grid(layout1,layout2)
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None]
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
    
    if Cp_op != 1:
        raise ValueError("Only option 1 is supported for Cp ")
    #power coefficient based on local inflow
    
    Uwt_ij = U_i[:,None]*(1-np.sum(deltaU_by_Uinf_f(r_ijk,theta_ijk,ct_ijk,K),axis=2)) #wake velocity at turbine locations
    pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(turb.Cp_f(Uwt_ij)*Uwt_ij**3))/(1*10**6)
    return pow_j,Uwt_ij

from utilities.helpers import gen_local_grid

def ntag_PA(Fourier_coeffs3_PA,
            layout1,
            layout2,
            turb,
            K,
            wav_Ct,
            RHO=1.225):
    #"No cross Term Analytical Gaussian" 
    # "Phase Amplitude (Fourier Series) form" (more computationally efficient!)
    #CT is constant across all wind directions :(
    A = turb.A

    r_jk,theta_jk = gen_local_grid(layout1,layout2)

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
        pow_j = (0.5*A*RHO*alpha)/(1*10**6)
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return pow_j,alpha

def caag_PA(cjd_noCp_PA_terms,
            layout1,
            layout2,
            turb,
            K,
            wav_Ct,
            Cp_op=1,WAV_CP=None,
            RHO=1.225):
    #"Cubed average analytical Gaussian" (the old way)
    # "Phase Amplitude (Fourier Series) form" (more computationally efficient!)
    #CT is constant across all wind directions :(
    A = turb.A

    r_jk,theta_jk = gen_local_grid(layout1,layout2)

    a_0,A_n,Phi_n = cjd_noCp_PA_terms

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
    if Cp_op == 1:
        Cp = turb.Cp_f(alpha)
    elif Cp_op == 2:
        if WAV_CP == None:
            raise ValueError("For option 2 provide WAV_CP")
        Cp = WAV_CP
    else:
        raise ValueError("No Cp option selected")

    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*A*RHO*Cp*alpha**3)/(1*10**6)
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return pow_j,alpha
