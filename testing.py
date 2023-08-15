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
def e(x,y,z=2):
    #some calculations
    return (x+y)*z

def f(*args,feedback=True,**kwargs):
    #some wrapper function
    if feedback:
        print("feedback")   
    return e(*args,**kwargs)

f(2,3,z=2,feedback="True")
