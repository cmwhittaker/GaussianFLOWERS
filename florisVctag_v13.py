#%% testing the new "form" of the formulation *now with complex exponentials!*
%load_ext autoreload
%autoreload 2

import numpy as np
no_xt = 2 #number of turbines in x
no_yt = no_xt
SITE = 6
LAYOUT_ROTATION = 22.5
SPACING = 5 #rectilinear spacing
K = 0.03
custom = None
RESOLUTION = 50

def rectangular_layout_rot(s=5,theta=45):
    xt = np.arange(1,no_xt+1,1)*s
    yt = np.arange(1,no_yt+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)
    xy = np.stack((Xt.reshape(-1), Yt.reshape(-1)), axis=1)
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    xt_rotated, yt_rotated = np.dot(rotation_matrix, xy.T)
    layout = np.column_stack((xt_rotated,yt_rotated))
    #xy_rotated = rotate(xy, angle=theta, reshape=False)

    return xt_rotated,yt_rotated,layout, np.size(Xt)#just a single layout for now

def rectangular_domain(layout,s=5,pad=3,r=200):
    Xt,Yt = layout[:,0],layout[:,1]
    pad = 1.0
    xr,yr = r,r #resolution
    X,Y = np.meshgrid(np.linspace(np.min(Xt)-pad*s,np.max(Xt)+pad*s,xr),np.linspace(np.min(Yt)-pad*s,np.max(Yt)+pad*s,yr))
    return X,Y,np.column_stack((X.reshape(-1),Y.reshape(-1)))

def calc_floris_powers(layout,wr_speed,wr_freq,D,wake=True):
    # a lil function to make the above more readable
    no_bins = wr_speed.size
    theta_i = np.linspace(0,360,no_bins,endpoint=False)
    Nt = layout.shape[0] #more readable
    pow_ij = np.zeros((no_bins,Nt))
    fi.reinitialize(layout_x=D*layout[:,0],layout_y=D*layout[:,1])
    for i in range(no_bins): #for each bin
        fi.reinitialize(
        #pretty silly
        wind_directions=np.array((theta_i[i],)),
        wind_speeds=np.array((wr_speed[i],)) #this will include the frequency
        )
        if wake == True:
            fi.calculate_wake() 
        else:
            fi.calculate_no_wake()
        
        pow_ij[i,:] = wr_freq[i]*fi.get_turbine_powers()/(1*10**6)
    pow_j = np.sum(pow_ij,axis=0)
    aep = np.sum(pow_ij)
    return pow_j,aep

from AEP3_2_functions import y_5MW
turb = y_5MW()
no_bins = 360
from distributions_vC05 import wind_rose
wr = wind_rose(bin_no_bins=no_bins,custom=custom,a_0=8,site=SITE,Cp_f=turb.Cp_f)
no_bins = wr.bin_no_bins

xt,yt,layout, Nt = rectangular_layout_rot(s=SPACING,theta=LAYOUT_ROTATION)
X,Y,plot_points = rectangular_domain(layout,s=SPACING,pad=3,r=RESOLUTION)

from AEP3_3_functions import ntag_v02,cubeAv_v4,ca_ag_v02,gen_local_grid_v01C,ntag_CE_v01

from floris.tools import FlorisInterface
fi = FlorisInterface("floris_settings.yaml")

# b0,c0 = calc_floris_powers(layout,
#                        wr.avMagnitude,
#                        wr.frequency,
#                        turb.D)

r_jk,theta_jk = gen_local_grid_v01C(layout,layout)

%timeit a1,b1,c1 = cubeAv_v4(r_jk,theta_jk,np.linspace(0,2*np.pi,no_bins,endpoint=False),wr.avMagnitude,wr.frequency,turb.Ct_f,turb.Cp_f,K,turb.A)

%timeit a2,b2,c2 = ntag_v02(r_jk,theta_jk,wr.cjd3_full_Fourier_coeffs,turb.Ct_f(np.sum(wr.frequency*wr.avMagnitude)),K,turb.A)

%timeit a3,b3,c3 = ca_ag_v02(r_jk,theta_jk,wr.cjd_full_Fourier_coeffs_noCp,turb.Cp_f,turb.Ct_f(np.sum(wr.frequency*wr.avMagnitude)),K,turb.A)

a_0,a_n,b_n = wr.cjd3_full_Fourier_coeffs

%timeit a6,b6,c6 = ntag_CE_v01(r_jk,theta_jk,np.copy(wr.cjd3_full_Fourier_coeffs_CMPLX),a_0,turb.Ct_f(np.sum(wr.frequency*wr. avMagnitude)),K,turb.A)

index = 3
print("a2[{}]/a6[{}]: {}".format(index,index,a6[index]/a2[index]))

#%%
data = ((wr.Cp*wr.frequency*(wr.avMagnitude**3)*wr.bin_no_bins)/(2*np.pi))
xs = np.linspace(0,2*np.pi,len(data),endpoint=False)
import scipy
d = scipy.fft.fft(data)/len(data)
d = scipy.fft.fftshift(d)

n2 = np.arange(-len(d)//2+1,len(d)//2+1,1)

f_reconstructed2 = np.sum(d[None,:]*np.exp(1j*n2[None,:]*xs[:,None]),axis=-1)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
ax.scatter(xs,data)
ax.scatter(xs,f_reconstructed2)

#%%

a6,b6,c6 = ntag_CE_v01(r_jk,theta_jk,np.copy(wr.cjd3_full_Fourier_coeffs_CMPLX),turb.Ct_f(np.sum(wr.frequency*wr. avMagnitude)),K,turb.A)
print("a6.dtype: {}".format(a6.dtype))
#%%
code_setup = '''pass'''

import timeit

NUMBER = 100
REPEAT =10
a1 = np.min(timeit.repeat(stmt=ntag_code_run,setup=code_setup, number=NUMBER,repeat=REPEAT))
a2 = np.min(timeit.repeat(stmt=ntag_code_run, setup=code_setup, number=NUMBER,repeat=REPEAT))
print("ntag: {}".format(a1))
print("ntag_CE: {}".format(a2))
print("ratio:{}".format(a1/a2))
#%%