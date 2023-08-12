#%% This aims to be a "definitive" location for all the functions used in "Phase 3" of the AEP project

def ntag_v01(layout,plot_points,cjd3_Fterms,CT,K,A,rho=1.225):
    #"No cross Term Analytical Gaussian" - the performance can probably be increased with a small amount of optimisation
    #CT is constant across all wind directions :(

    xt_j,yt_j = layout[:,0],layout[:,1]
    xp_j,yp_j = plot_points[:,0],plot_points[:,1]

    x_jk = xp_j[:, None] - xt_j[None, :]
    y_jk = yp_j[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    theta_jk = np.pi/2 - np.arctan2(y_jk, x_jk)
    #coord system conversion

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
        cnst_term = (np.sqrt(2*np.pi*a)*sigma/(2*a*np.pi))*(sqrt_term**a)
        mfs = (a_0/2 + np.sum(np.exp(-((sigma_b*n_b)**2)/(2*a))*(a_n*np.cos(n_b*theta_b)+b_n*np.sin(n_b*theta_b)),axis=-1)) #modified Fourier series
        return np.sum(cnst_term*mfs,axis=-1)
    
    #alpha is the 'energy' content of the wind
    #I don't know why this 2pi is needed, *but it is*
    alpha = (a_0/2 - 3*term(1) + 3*term(2) - term(3))*2*np.pi
    if len(xt_j) == len(xp_j): #farm aep calculation
        pow_j = (0.5*A*rho*alpha)/(1*10**6)
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return alpha,pow_j,aep


def cubeAv_v3(layout,plot_points,theta_i,U_i,P_i,ct_f,cp_f,K,A,rho=1.225):
    #calculates the (average) wake velocity and farm aep discretely
    def deltaU_by_Uinf(r,theta,ct,K):
        theta = theta + np.pi #wake is downstream
        ep = 0.2*np.sqrt((1+np.sqrt(1-ct))/(2*np.sqrt(1-ct)))

        U_delta_by_U_inf = (1-np.sqrt(1-(ct/(8*(K*r*np.sin(theta)+ep)**2))))*(np.exp(-(r*np.cos(theta))**2/(2*(K*r*np.sin(theta)+ep)**2)))

        lim = (np.sqrt(ct/8)-ep)/K #this is the y value of the invalid region, can be negative depending on Ct
        lim = np.where(lim<0.01,0.01,lim) #may sure it's always atleast 0.01 (stop self-produced wake) (this should be >0 but there is numerical artifacting in rsin(theta) )
        deltaU_by_Uinf = np.where(r*np.sin(theta)>lim,U_delta_by_U_inf,0) #this stops turbines producing their own deficit 
        return deltaU_by_Uinf

    #I sometimes use this function to find the wake layout, so find relative posistions to plot points not the layout 
    #when plot_points = layout it finds wake at the turbine posistions
    xt_j,yt_j = layout[:,0],layout[:,1]
    xp_j,yp_j = plot_points[:,0],plot_points[:,1]

    x_jk = xp_j[:, None] - xt_j[None, :]
    y_jk = yp_j[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    theta_jk = np.arctan2(y_jk, x_jk)

    theta_ijk = theta_jk[None,:,:] + theta_i[:,None,None]
    r_ijk = np.repeat(r_jk[None,:,:],len(theta_i),axis=0)
    ct_ijk = ct_f(U_i)[...,None,None]*np.ones((len(xp_j),len(xt_j)))[None,...] #this is a dirty way of repeating along 2 axis
    Uw_ij = U_i[:,None]*(1-np.sum(deltaU_by_Uinf(r_ijk,theta_ijk,ct_ijk,K),axis=2))
    if len(xt_j) == len(xp_j): #farm aep calculation
        pow_ij = P_i[:,None]*(0.5*A*rho*cp_f(Uw_ij)*Uw_ij**3)/(1*10**6) #this IS slower, but needed for illustrations
        aep = np.sum(pow_ij)
    else: #farm wake visualisation
        pow_ij = np.nan
        aep = np.nan
        Uw_ij = np.sum(P_i[:,None]*Uw_ij,axis=0)
    return Uw_ij,np.sum(pow_ij,axis=0),aep

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import numpy as np

import numpy as np

def ca_ag_v01(layout,plot_points,cjd_Fterms,Cp_f,CT,K,A,rho=1.225):
    #"Cubed Average Analytical Gaussian"

    xt_j,yt_j = layout[:,0],layout[:,1]
    xp_j,yp_j = plot_points[:,0],plot_points[:,1]

    x_jk = xp_j[:, None] - xt_j[None, :]
    y_jk = yp_j[:, None] - yt_j[None, :]

    r_jk = np.sqrt(x_jk**2+y_jk**2)
    theta_jk = np.pi/2 - np.arctan2(y_jk, x_jk)
    #coord system conversion

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

    if len(xt_j) == len(xp_j): #farm aep calculation
        pow_j = (0.5*A*rho*Cp_f(alpha)*alpha**3)/(1*10**6)
        #per-turbine power generation
        aep = np.sum(pow_j)
    else: #farm wake visualisation
        pow_j = np.nan
        aep = np.nan
    return alpha,pow_j,aep

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

class y_5MW():
    def __init__(self):
        import numpy as np
        self.Cp = np.array([0. , 0. , 0.074 , 0.3251 , 0.3762 , 0.4027 , 0.4156 , 0.423 , 0.4274 , 0.4293 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.4298 , 0.429603, 0.354604, 0.316305, 0.281478, 0.250068, 0.221924, 0.196845, 0.174592, 0.154919, 0.13757 , 0.1223 , 0.108881, 0.097094, 0.086747, 0.077664, 0.069686, 0.062677, 0.056511, 0.051083, 0.046299, 0.043182, 0.033935, 0. , 0. ])

        self.Ct = np.array([0. , 0. , 0.7701, 0.7701, 0.7763, 0.7824, 0.782 , 0.7802, 0.7772, 0.7719, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7768, 0.7675, 0.7651, 0.7587, 0.5056, 0.431 , 0.3708, 0.3209, 0.2788, 0.2432, 0.2128, 0.1868, 0.1645, 0.1454, 0.1289, 0.1147, 0.1024, 0.0918, 0.0825, 0.0745, 0.0675, 0.0613, 0.0559, 0.0512, 0.047 , 0. , 0. ])

        self.wind_speed = np.array([ 0. , 2.9 , 3. , 4. , 4.5147, 5.0008, 5.4574, 5.8833, 6.2777, 6.6397, 6.9684, 7.2632, 7.5234, 7.7484, 7.9377, 8.0909, 8.2077, 8.2877, 8.3308, 8.337 , 8.3678, 8.4356, 8.5401, 8.6812, 8.8585, 9.0717, 9.3202, 9.6035, 9.921 , 10.272 , 10.6557, 11.5077, 12.2677, 12.7441, 13.2494, 13.7824, 14.342 , 14.9269, 15.5359, 16.1675, 16.8204, 17.4932, 18.1842, 18.8921, 19.6152, 20.3519, 21.1006, 21.8596, 22.6273, 23.4019, 24.1817, 24.75 , 25.01 , 25.02 , 50. ])

        self.Z_h = 90 #height
        self.D = 126 #diameter
        self.A = np.pi*(self.D/2)**2 #area

        self.Cp_f = lambda u: self.Cp_itrp(u)
        self.Ct_f = lambda u: self.Ct_itrp(u)

    def Ct_itrp(self,u):
        return np.interp(u,self.wind_speed,self.Ct)

    def Cp_itrp(self,u):
        return np.interp(u,self.wind_speed,self.Cp)  
    
