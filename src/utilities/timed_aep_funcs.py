#%% This are all most of the AEP functions wrapped in functions that time their execution using %timeit 
# The %command's are interactive python line magic commands, so I think you need Ipython installed to run them
#% times = %timeit -o -q f(x) means: time f(x),(-o) save result as times object, (-q) without feedback. Should take about 7-14 seconds

import numpy as np
import time


def slow_numerical_aep(U_i,P_i,theta_i,layout,plot_points,turb,K):
    #this finds the thrust coefficient "in turn", the calculation can't be vectorised so its the slow way.
    #This is only useful as an accuracy comparison, so it's not timed
    from AEP3_functions import num_F_v02
    pow_j,_,Uwff_ja= num_F_v02(U_i,P_i,theta_i,
                     layout,
                     plot_points,
                     turb,
                     K,
                     u_lim=None,
                     Ct_op=1,
                     Cp_op=1,
                     cross_ts=True,ex=False)
    return pow_j,Uwff_ja

def fast_numerical_aep(U_i,P_i,theta_i,layout,turb,K,timed=True):
    #this is the "numerical integration" using a global thrust coefficient (Ct_op=2) but local power coefficient (Ct_op = 1). The small angle approximation has a negligible effect on accuracy but increases performance, so I use it for a more accurate performance comparsion.
    from AEP3_functions import cubeAv_v5
    if timed:
        result = []
        timings = %timeit -o -q result.append(cubeAv_v5(U_i,P_i,theta_i,layout,layout,turb,K,u_lim=None,Ct_op=2,Cp_op=1,ex=False))
        pow_j,_ = result[0]
        # this is a workaround to get the tuple result out properly
        time = timings.best
    else:
        pow_j = cubeAv_v5(U_i,P_i,theta_i,layout,layout,turb,K,u_lim=None,Ct_op=2,Cp_op=1,ex=False)
        time = np.NaN
    
    return pow_j,time

def ntag_timed_aep(Fourier_coeffs3_PA,layout,turb,K,wav_Ct,timed=True):
    #No cross Terms analytical Gaussian (Gaussian Flowers) approach
    from AEP3_functions import ntag_PA_v03
    if timed:
        result = []
        timings = %timeit -o -q result.append(ntag_PA_v03(Fourier_coeffs3_PA,layout,layout,turb,K,wav_Ct))
        pow_j,_ = result[0]
        # this is a workaround to get the tuple result out properly
        time = timings.best
    else:
        pow_j,_ = ntag_PA_v03(Fourier_coeffs3_PA,layout,layout,turb,K,wav_Ct)
        time = np.NaN
    return pow_j,time

def caag_timed_aep(Fourier_coeffs_noCp_PA,layout,turb,K,wav_Ct,timed=True):
    #Cube of the Average Analytical Gaussian (this )
    from AEP3_functions import caag_PA_v03
    if timed:
        result = []
        timings = %timeit -o -q result.append(caag_PA_v03(Fourier_coeffs_noCp_PA,layout,layout,turb,K,wav_Ct))
        pow_j,_ = result[0]
        # this is a workaround to get the tuple result out properly
        time = timings.best
    else:
        pow_j,_ = caag_PA_v03(Fourier_coeffs_noCp_PA,layout,layout,turb,K,wav_Ct)
        time = np.NaN
    return pow_j,time