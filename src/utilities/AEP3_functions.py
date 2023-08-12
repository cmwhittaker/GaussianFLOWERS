#%% These are all the wake calculating functions 
import numpy as np

def ntag_v02(r_jk,theta_jk,cjd3_Fterms,CT,K,A,rho=1.225):
    #"No cross Term Analytical Gaussian" - the performance can probably be increased with a small amount of optimisation
    #CT is constant across all wind directions :(

    a_0,a_n,b_n = cjd3_Fterms

    EP = 0.2*np.sqrt((1+np.sqrt(1-CT))/(2*np.sqrt(1-CT)))

    #auxilaries
    n = np.arange(1,a_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(CT/(8*(K*r_jk+EP)**2)))))

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    a_n = a_n[None,None,:]
    b_n = b_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream

    def term(a):
        cnst_term = ((np.sqrt(2*np.pi*a)*sigma)/(a))*(sqrt_term**a)
        mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2*a))*(a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)
    
    #alpha is the 'energy' content of the wind
    alpha = (a_0/2)*2*np.pi - 3*term(1) + 3*term(2) - term(3)
    
    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*A*rho*alpha)/(1*10**6)
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return alpha,pow_j,aep

def ntag_PA_v02(r_jk,theta_jk,cjd3_PA_terms,WAV_CT,K,A,rho=1.225):
    #"No cross Term Analytical Gaussian" 
    # "Phase Amplitude (Fourier Series) form" (more computationally efficient!)
    #CT is constant across all wind directions :(

    a_0,A_n,Phi_n = cjd3_PA_terms

    EP = 0.2*np.sqrt((1+np.sqrt(1-WAV_CT))/(2*np.sqrt(1-WAV_CT)))

    #auxilaries
    n = np.arange(1,A_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(WAV_CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(WAV_CT/(8*(K*r_jk+EP)**2)))))

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    A_n = A_n[None,None,:]
    Phi_n = Phi_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream

    def term(a):
        cnst_term = ((np.sqrt(2*np.pi*a)*sigma)/(a))*(sqrt_term**a)
        mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2*a))*(A_n*np.cos(n_b*theta_b+Phi_n)),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)
    
    #alpha is the 'energy' content of the wind
    alpha = (a_0/2)*2*np.pi - 3*term(1) + 3*term(2) - term(3)
    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*A*rho*alpha)/(1*10**6)
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return alpha,pow_j,aep

def ntag_CE_v01(r_jk,theta_jk,c_n,a_0,CT,K,A,rho=1.225):
    #"No cross Term Analytical Gaussian" -
    #"COMPLEX EXPONENTIAL" form
    # the performance can probably be increased with a small amount of optimisation
    #CT is constant across all wind directions :(

    EP = 0.2*np.sqrt((1+np.sqrt(1-CT))/(2*np.sqrt(1-CT)))
    
    if len(c_n)%2 == 0: #EVEN
        n = np.arange(-len(c_n)/2,len(c_n)//2,1)
    else: #ODD
        n = np.arange(-(len(c_n)-1)/2,(len(c_n)-1)/2+1,1)

    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(CT/(8*(K*r_jk+EP)**2)))))

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    c_n = c_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream

    def term(a):
        cnst_term = (np.sqrt(2*np.pi*a)*sigma/(a))*(sqrt_term**a)
        arg = -(sigma_b*n_b)**2/(2*a) +1j*n_b*theta_b
        mfs = np.sum(c_n * np.exp(arg),axis=-1).astype('float64')
        return np.sum(cnst_term*mfs,axis=-1)
    
    #alpha is the 'energy' content of the wind
    alpha = (a_0/2)*2*np.pi - 3*term(1) + 3*term(2) - term(3)
    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*A*rho*alpha)/(1*10**6)
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return alpha,pow_j,aep

def cubeAv_v4(r_jk,theta_jk,theta_i,U_i,P_i,ct_f,cp_f,K,A,rho=1.225):
    #(DEPRECIATED!)
    #Pretty sure this is wrong ...
    #(or atleast I can't properly verify it!)
    #calculates the (average) wake velocity and farm aep discretely
    def deltaU_by_Uinf(r,theta,ct,K):
        ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))

        U_delta_by_U_inf = (1-np.sqrt(1-(ct/(8*(K*r*np.sin(theta)+ep)**2))))*(np.exp(-(r*np.cos(theta))**2/(2*(K*r*np.sin(theta)+ep)**2)))

        lim = (np.sqrt(ct/8)-ep)/K #this is the y value of the invalid region, can be negative depending on Ct
        lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be >0 but there is numerical artifacting in rsin(theta) )
        deltaU_by_Uinf = np.where(r*np.sin(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
        return deltaU_by_Uinf

    #I sometimes use this function to find the wake layout, so find relative posistions to plot points not the layout 
    #when plot_points = layout it finds wake at the turbine posistions
    theta_ijk = theta_jk[None,:,:] - theta_i[:,None,None] + 3*np.pi/2 # I don't know

    r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)
    ct_ijk = ct_f(U_i)[...,None,None]*np.ones((r_jk.shape[0],r_jk.shape[1]))[None,...] #bad way of repeating
    Uw_ij = U_i[:,None]*(1-np.sum(deltaU_by_Uinf(r_ijk,theta_ijk,ct_ijk,K),axis=2))
    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_ij = P_i[:,None]*(0.5*A*rho*cp_f(Uw_ij)*Uw_ij**3)/(1*10**6) #this IS slower, but needed for illustrations
        aep = np.sum(pow_ij)
        flow_field = np.nan
    else: #farm wake visualisation
        pow_ij = np.nan
        aep = np.nan
        flow_field = np.sum(cp_f(Uw_ij)*P_i[:,None]*Uw_ij**3,axis=0)
    return flow_field,np.sum(pow_ij,axis=0),aep

def ca_ag_v02(r_jk,theta_jk,cjd_Fterms_noCP,Cp_f,CT,K,A,rho=1.225):
    
    #"Cubed Average Analytical Gaussian"

    a_0,a_n,b_n = cjd_Fterms_noCP

    EP = 0.2*np.sqrt((1+np.sqrt(1-CT))/(2*np.sqrt(1-CT)))

    #auxilaries
    n = np.arange(1,a_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim) #limit always posistive
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(CT/(8*(K*r_jk+EP)**2)))))
    cnst_term = sqrt_term*np.sqrt(2*np.pi)*sigma

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]    
    sigma_b = sigma[:,:,None]
    a_n = a_n[None,None,:]
    b_n = b_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream

    mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2))*(a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)),axis=-1)) #modified Fourier series
    alpha = 2*np.pi*a_0/2 - np.sum(cnst_term*mfs,axis=-1)  
    #alpha is the linear wake velocity

    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*A*rho*Cp_f(alpha)*alpha**3)/(1*10**6)
        #per-turbine power generation
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return alpha,pow_j,aep

def ca_cp_ag_v01(r_jk,theta_jk,cjd_Fterms,Cp_f,CT,K,A,rho=1.225):
    # (effectively depreciated)

    #"Cubed Average Analytical Gaussian"
    #INCLUDING the power coefficient

    #This approximation doesn't work very well!

    a_0,a_n,b_n = cjd_Fterms

    EP = 0.2*np.sqrt((1+np.sqrt(1-CT))/(2*np.sqrt(1-CT)))

    #auxilaries
    n = np.arange(1,a_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim) #limit always posistive
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(CT/(8*(K*r_jk+EP)**2)))))
    cnst_term = sqrt_term*np.sqrt(2*np.pi)*sigma

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]    
    sigma_b = sigma[:,:,None]
    a_n = a_n[None,None,:]
    b_n = b_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream

    mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2))*(a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)),axis=-1)) #modified Fourier series
    alpha = 2*np.pi*a_0/2 - np.sum(cnst_term*mfs,axis=-1)  
    #alpha is the linear wake velocity
    
    print("alpha: {}".format(alpha))

    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*A*rho*alpha**3)/(1*10**6)
        #per-turbine power generation
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return alpha,pow_j,aep

def num_F_v02(U_i,P_i,theta_i,
              layout,
              plot_points, #this is the comp domain
              turb,
              RHO=1.225,K=0.025,
              u_lim=None,Ct_op=True,WAV_CT=None,cross_ts=True,ex=True,Cp_op=True,WAV_CP=None,cube_term=True):
    #version 2! 
    #this now finds the wakes of the turbines "in turn" so that it can support a thrust coefficient based on local inflow: Ct(U_w) not Ct(U_\infty) as previously.
    if np.any(np.abs(theta_i) > 10): #this is needed ...
        raise ValueError("Did you give num_F_v02 degrees?")
    
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
    
    def soat(a): #Sum over Axis Two
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

from utilities.helpers import gen_local_grid

def ntag_PA_v03(cjd3_PA_terms,layout1,layout2,turb,WAV_CT,K,RHO=1.225):
    #"No cross Term Analytical Gaussian" 
    # "Phase Amplitude (Fourier Series) form" (more computationally efficient!)
    #CT is constant across all wind directions :(
    A = turb.A

    r_jk,theta_jk = gen_local_grid(layout1,layout2)

    a_0,A_n,Phi_n = cjd3_PA_terms

    EP = 0.2*np.sqrt((1+np.sqrt(1-WAV_CT))/(2*np.sqrt(1-WAV_CT)))

    #auxilaries
    n = np.arange(1,A_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(WAV_CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(WAV_CT/(8*(K*r_jk+EP)**2)))))

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

def caag_PA_v03(cjd_noCp_PA_terms,layout1,layout2,turb,WAV_CT,K,Cp_op=1,WAV_CP=None,RHO=1.225):
    #"Cubed average analytical Gaussian" (the old way)
    # "Phase Amplitude (Fourier Series) form" (more computationally efficient!)
    #CT is constant across all wind directions :(
    A = turb.A

    r_jk,theta_jk = gen_local_grid(layout1,layout2)

    a_0,A_n,Phi_n = cjd_noCp_PA_terms

    EP = 0.2*np.sqrt((1+np.sqrt(1-WAV_CT))/(2*np.sqrt(1-WAV_CT)))

    #auxilaries
    n = np.arange(1,A_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(WAV_CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(WAV_CT/(8*(K*r_jk+EP)**2)))))
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

def cubeAv_v5(U_i,P_i,theta_i,
              layout1,
              layout2, 
              turb,
              RHO=1.225,K=0.025,
              u_lim=None,ex=True,Ct_op=1,WAV_CT=None,Cp_op=1):
    #discrete numerical convolution
    # effectively num_F_v02 with 
    # OPTIONS:
    # Ct_op = 1 or 2 
    # ex=True or False
    # u_lim 
    # FIXED
    # cross_ts=True
    # Cp_op = 2

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
        raise ValueError("Local Ct is not supported by this function")
    elif Ct_op == 2: #base global inflow
        ct_ijk = np.broadcast_to(turb.Ct_f(U_i)[...,None,None],r_ijk.shape)
    elif Ct_op == 3:
        if WAV_CT == None:
            raise ValueError("For option 3 provide WAV_CT")
        ct_ijk = np.broadcast_to(WAV_CT,r_ijk.shape)
    else:
        raise ValueError("No Ct option selected")
    
    if Cp_op != 1:
        raise ValueError("Only option 1 is supported for Cp ")
    
    Uwt_ij = U_i[:,None]*(1-np.sum(deltaU_by_Uinf_f(r_ijk,theta_ijk,ct_ijk,K),axis=2))
    pow_j = 0.5*turb.A*RHO*np.sum(P_i[:,None]*(turb.Cp_f(Uwt_ij)*Uwt_ij**3))/(1*10**6)
    return pow_j,Uwt_ij

def ntag_PA_TS_v01(fc_PA_a,fc_PA_b,m,c,layout1,layout2,turb,WAV_CT,K,RHO=1.225):
    #"No cross Term Analytical Gaussian" 
    # "Phase Amplitude (Fourier Series) form" (more computationally efficient!)
    #CT is constant across all wind directions :(
    r_jk,theta_jk = gen_local_grid(layout1,layout2)

    a_0,A_n,Phi_n = fc_PA_a
    EP = 0.2*np.sqrt((1+np.sqrt(1-WAV_CT))/(2*np.sqrt(1-WAV_CT)))

    #auxilaries
    n = np.arange(1,A_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(WAV_CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(WAV_CT/(8*(K*r_jk+EP)**2)))))

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    A_n = A_n[None,None,:]
    Phi_n = Phi_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream
    
    #auxilaries

    def term(a):
        cnst_term = ((np.sqrt(2*np.pi*a)*sigma)/(a))*(sqrt_term**a)
        mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2*a))*(A_n*np.cos(n_b*theta_b+Phi_n)),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)

    alpha1 = m*((a_0/2)*2*np.pi - 4*term(1) + 6*term(2) - 4*term(3) + term(4))

    a_0,A_n,Phi_n = fc_PA_b
    EP = 0.2*np.sqrt((1+np.sqrt(1-WAV_CT))/(2*np.sqrt(1-WAV_CT)))

    #auxilaries
    n = np.arange(1,A_n.size+1,1)
    sigma = np.where(r_jk!=0,(K*r_jk+EP)/r_jk,0)
    lim = (np.sqrt(WAV_CT/8)-EP)/K
    lim = np.where(lim<0.01,0.01,lim)
    sqrt_term = np.where(r_jk<lim,0,(1-np.sqrt(1-(WAV_CT/(8*(K*r_jk+EP)**2)))))

    #modify some dimensions ready for broadcasting
    n_b = n[None,None,:]  
    sigma_b = sigma[:,:,None]
    A_n = A_n[None,None,:]
    Phi_n = Phi_n[None,None,:]
    theta_b = theta_jk[:,:,None] + np.pi #wake is downstream

    def term(a):
        cnst_term = ((np.sqrt(2*np.pi*a)*sigma)/(a))*(sqrt_term**a)
        mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2*a))*(A_n*np.cos(n_b*theta_b+Phi_n)),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)

    alpha2 = c*((a_0/2)*2*np.pi - 3*term(1) + 3*term(2) - term(3))
    A = turb.A
    alpha = alpha1+alpha2
    if r_jk.shape[0] == r_jk.shape[1]: #farm aep calculation
        pow_j = (0.5*A*RHO*alpha)/(1*10**6)
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return pow_j,alpha
