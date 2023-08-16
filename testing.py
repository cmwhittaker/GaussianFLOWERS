#%% general testing script

from src.utilities.turbines import iea_15MW
turb = iea_15MW()

#%% 
import numpy as np
from src.utilities.helpers import rectangular_layout
layout = rectangular_layout(3,4,np.deg2rad(10))
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=200)
ax.set(aspect='equal')
ax.scatter(layout[:,0],layout[:,1])

#%% relearning the timeit so I can remove the interactive python line magic

import time
def f(x,y):
    time.sleep(1) #mimick some computation time
    c = x**2+y
    return None
x,y = 2,3
%timeit f(x,y)

#%% learning unpacking
def e(x,y,z=2,w=None):
    #some calculations
    return x+y+z+w

def f(*args,feedback=True,**kwargs):
    #some wrapper functionw
    if feedback:
        print("feedback")   
    return e(*args,w=1,**kwargs)

f(1,1,z=0,feedback=True)


#%% 
def function(flag=True):
    if flag:
        return 1
    return 2

print(function(flag=False))

#%%

list = [1,2,3,4]
for a in list:
    print()

#%%
def f(x):
    print("f has been run")
    return None
g = lambda: f(1)
