#%% Investigating the performance cost of calculations using python's timeit

import timeit
import time

code_run = 'f(1)'

code_setup = '''
def f(x):
    time.sleep(1)
    return None
'''
start_time = time.time()
a = timeit.repeat(stmt=code_run, setup=code_setup, number=2,repeat=3)
print(min(a))
delta_time = time.time()-start_time
print("time taken: {}".format(delta_time))

#%%
code_run = 'f(1.234+1j)'

code_setup = '''
import numpy as np
def f(x): 
    return np.real(x)
'''
start_time = time.time()
NUMBER = 10000
a = timeit.repeat(stmt=code_run, setup=code_setup, number=NUMBER,repeat=1000)
print(min(a)/NUMBER)
delta_time = time.time()-start_time
print("time taken: {}".format(delta_time))

#%% 
# comparing the performance difference in the complex vs. "normal" way of reconstructing the fourier series
import numpy as np
SITE = 7
from distributions_vC05 import wind_rose
from AEP3_2_functions import y_5MW
turb = y_5MW()
no_bins = 72
wr = wind_rose(bin_no_bins=no_bins,custom=None,a_0=8,site=SITE,Cp_f=turb.Cp_f)

c_n = wr.cjd3_full_Fourier_coeffs_CMPLX
c_n[0] = 0.5*c_n[0]
a_0,a_n,b_n = wr.cjd3_full_Fourier_coeffs

n1 = np.arange(0,len(c_n),1)
n2 = np.arange(1,len(a_n)+1,1) #n array for broadcasting
n3 = np.arange(-len(c_n)+1,len(c_n),1) #n array for broadcasting

actual = (wr.Cp*wr.frequency*(wr.avMagnitude**3)*wr.bin_no_bins)/(2*np.pi)

xs = np.linspace(0,2*np.pi,72,endpoint=False)

a1 = np.sum(c_n[None,:]*np.exp(1j*n1[None,:]*xs[:,None]),axis=-1).astype('float64')

a2 = a_0/2 + np.sum((a_n[None,:]*np.cos(n2[None,:]*xs[:,None])+b_n[None,:]*np.sin(n2[None,:]*xs[:,None])),axis=-1)

print("a1[0]: {}".format(a1[0]))
print("a2[0]: {}".format(a2[0]))

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
ax.scatter(xs,actual,color='g',marker='+',s=100)
ax.scatter(xs,a1,color='orange',marker='x',s=100)
ax.scatter(xs,a2,color='red',marker='2',s=100)
