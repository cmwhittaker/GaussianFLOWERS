import numpy as np

def spacial_average(U,x,y,z,xt,yt,zt,d,pts=6):
    #function to find the average inlet velocity over the "disk" of the turbine in the velocity field (U)

    #auxilary constants
    r = d/2 
    XP,YP,ZP = np.meshgrid(xt,y,z,indexing='ij')
    b = np.sqrt((YP-yt)**2+(ZP-zt)**2)<r
    XP,YP,ZP = XP[b],YP[b],ZP[b] #remove points outside of disk

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((x,y,z), U) 

    pts = np.hstack((XP.reshape((-1,1)),YP.reshape((-1,1)),ZP.reshape(-1,1))) #points to sample at

    if pts.shape[0] < 3: #warning for stupid situations
        raise UserWarning("less than 3 points sampled!")

    return np.mean(interp(pts)) #possibly redo this later to specific which axis to take the mean over (to support "full" vectorisation)

def intersection_frac(yn,zn,rn,yi,zi,ri):
    #find the intersection area (if there is any) between two circles
    #it is assumed that ri > rn
    #works when xi is a 1d np array

    r = rn 
    R = ri
    d = np.sqrt((yi-yn)**2+(zi-zn)**2)

    if np.any(R<r):
        raise ValueError("R!<r care")

    with np.errstate(all='ignore'):
        part1 = r*r*np.arccos((d*d + r*r - R*R)/(2*d*r))
        part2 = R*R*np.arccos((d*d + R*R - r*r)/(2*d*R))
        part3 = 0.5*np.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R))
        intersectionArea = part1 + part2 - part3

    #nested (3-way) np.where. d+r<=R: no overlap, d+r<=R: fully contained, d+r<=R: what's left (partial overlap)
    frac = np.where(d+r<=R,1,np.where(d>=r+R,0,intersectionArea))
  
    return frac

def wake_added_turb(x_n,y_n,z_n,x_i,y_i,z_i,k_i,Ct_i,I0,D):
    a=.5*(1-np.sqrt(1-Ct_i)) #this will be for all upstream turbines
    Ip_n=.65*(a**(.83))*(I0**.03)*((x_n-x_i)/D)**(-.32)
    r_i = 3*(k_i*(x_n-x_i)+epsilon(Ct_i)*D)
    r_n = D/2
    I_add = np.max(Ip_n*intersection_frac(y_n,z_n,r_n,y_i,z_i,r_i))
    return I_add #return the maximum wake-added turbulence

def epsilon(Ct): #initial wake width is a function of Ct only
    return 0.2*np.sqrt((1+np.sqrt(1-Ct))/(2*np.sqrt(1-Ct)))
            
    
class NREL5W():

    def __init__(self):

        self.speed=np.array((2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5))

        self.thrust=np.array((1.4612728464576872, 1.3891500248600195, 1.268082754962957, 1.1646999475504172, 1.0793803926905128, 1.0098020917279509, 0.9523253671258429, 0.9048200632193146, 0.8652746358037285, 0.8317749797630494, 0.8032514305647592, 0.7788892341777304, 0.7730863447173755, 0.7726206761501038, 0.7721934195205071, 0.7628473779358198, 0.7459330274762097, 0.7310049480450205, 0.7177914274917664, 0.799361832581412, 0.8871279360742889, 0.9504655842078242, 1.0000251651970853, 1.0390424010487957, 1.0701572223736, 1.0945877239199593))

        self.power=np.array((-0.2092219804533027, 0.2352391893638198, 0.46214453324002824, 0.5476677311380832, 0.5772456648046942, 0.5833965967255043, 0.5790298877294793, 0.5701467792599509, 0.5595564940228319, 0.5480479331210222, 0.5366246493538858, 0.5258303873334416, 0.5229191014420005, 0.5224657416437077, 0.5220516710065948, 0.5175531496262384, 0.5092952304943719, 0.5016730194861562, 0.4946298748497652, 0.5326349577484786, 0.5597671514540806, 0.5679550280111124, 0.5659876382489049, 0.5572755521043566, 0.5441595739848516, 0.5280326705762761))

        return None
        
    def C_t(self,u):
        Ct = np.interp(u,self.speed,self.thrust)
        if np.any(Ct > 1):
            print("Ct is greater than 1, careful!")
        return Ct

    def C_p(self,u):
        return np.interp(u,self.speed,self.power)

class VestaV80():
    def __init__(self):
        self.CTSPEED = np.array([0.,5.,5.02545,5.24604,5.36482,5.61934,5.806,6.1793,6.38292,6.73925,6.95984,7.31618,7.53676,7.87613,8.09672,8.43609,8.65667,9.01301,9.2336,9.58993,9.77658,10.09898,10.2517,10.38744,10.50622,10.72681,11.06618,11.28676,11.59219,11.71097,11.93156,12.03337,12.15215,12.20305,12.32183,12.38971,12.49152,12.55939,12.6612,12.74604,12.84785,12.93269,13.05147,13.17025,13.33993,13.45871,13.62839,13.74717,13.95079,14.08654,14.29016,14.44287,14.68043,14.83314,15.08767,15.24038,15.35916,15.56278,15.7155,15.98699,16.17364,16.47907,16.69966,16.97115,17.20871,17.54808,17.7517,18.0741,18.31165,18.65102,18.87161,19.21097,19.48247,19.7879,19.97455,20.,30.])
        self.CTCT = np.array([0.,0.,0.80687,0.80557,0.80428,0.80428,0.80428,0.80428,0.80428,0.80428,0.80428,0.80557,0.80557,0.80687,0.80687,0.80687,0.80687,0.80428,0.80298,0.80039,0.7965,0.78483,0.77447,0.7641,0.75632,0.75243,0.74595,0.74076,0.7278,0.71743,0.6954,0.68244,0.65522,0.64226,0.61504,0.59819,0.57356,0.55671,0.53338,0.51523,0.4919,0.47375,0.45172,0.43616,0.41413,0.39728,0.37524,0.36228,0.34025,0.32599,0.30525,0.29229,0.27285,0.26118,0.24174,0.23137,0.22359,0.21452,0.20544,0.18989,0.18211,0.17045,0.16267,0.1536,0.14712,0.13804,0.13415,0.13027,0.12508,0.1199,0.1173,0.11471,0.10823,0.10564,0.10434,0.,0.])
        self.CPSPEED = np.array([0.,5.,5.00848,5.17817,5.36482,5.56844,5.75509,5.89084,6.07749,6.24717,6.41686,6.56957,6.77319,7.01075,6.89197,7.16346,7.29921,7.41799,7.53676,7.72342,7.85916,7.97794,8.09672,8.2155,8.35124,8.48699,8.60577,8.72455,8.84333,8.97907,9.09785,9.21663,9.31844,9.42025,9.52206,9.62387,9.75962,9.86143,9.96324,10.06505,10.18382,10.28563,10.42138,10.52319,10.69287,10.87952,10.9983,11.16799,11.33767,11.52432,11.74491,11.89762,12.08428,12.2879,12.49152,12.67817,12.91572,13.22115,13.44174,13.7302,14.01867,14.30713,14.54468,14.86708,15.00283,16.7336,18.22681,19.99152,20.,30.])
        self.CPCP = np.array([0.,0.,0.16202,0.18017,0.20091,0.22683,0.24757,0.2709,0.29682,0.32534,0.35386,0.37978,0.41866,0.46792,0.4394,0.49644,0.52754,0.55347,0.58717,0.62865,0.66753,0.69605,0.72975,0.76863,0.80493,0.84381,0.88529,0.92158,0.96047,0.99935,1.03824,1.07712,1.11082,1.14712,1.18082,1.21711,1.25859,1.29229,1.3234,1.35969,1.39857,1.42968,1.47116,1.50227,1.55412,1.60855,1.64226,1.68373,1.7278,1.77187,1.81335,1.84705,1.87557,1.90149,1.92223,1.93778,1.95593,1.96371,1.96889,1.96889,1.97148,1.97926,1.98704,1.99482,2.,2.,2.,2.,0.,0.])
        self.U_h = 70
        self.d = 80

    def C_t(self,u):
        return np.interp(u,self.CTSPEED,self.CTCT)

    def C_p(self,u):
        return np.interp(u,self.CPSPEED,self.CPCP)   

def get_Horns_layout(): #0,0 is bottom left
    v_inter = 7*np.cos(np.deg2rad(7))
    h_inter = 7
    yt = np.arange(0,8*v_inter,v_inter)
    xt = np.arange(0,10*h_inter,h_inter)
    Xt,Yt = np.meshgrid(xt,yt)
    Xt = Xt - Yt*np.sin(np.deg2rad(7)) #add skew
    return Xt,Yt