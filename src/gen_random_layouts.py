#%% a script to generate some "random" layouts and pickle them so I can use them later
import sys
if hasattr(sys, 'ps1'):
    #if it's interactive, re-import modules every run
    %load_ext autoreload
    %autoreload 2

import numpy as np
N_LAYOUTS = 10
np.random.seed(1)

from utilities.poissonDiscSampler import PoissonDisc

layouts = []
n_turbs = []
density = []
pd = 0.46 #empirically found packing density
for i in range(N_LAYOUTS):
    r = np.random.uniform(3.7, 7)
    n = np.random.uniform(5, 60)
    w = np.sqrt((n*np.pi*r**2)/(4*pd))-2*r
    #print(f"r/n/w:{r}/{n}/{w}")
    sampler = PoissonDisc(w,w,r)
    coords = sampler.sample()
    layout = np.asarray(coords)
    layout = layout - w/2
    layouts.append(layout)
    n_turbs.append(len(layout))

np.save('rdm_layouts.npy', layouts)

#%% layout reference
from matplotlib.gridspec import GridSpec 
gs = GridSpec(5, 5)
fig = plt.figure(figsize=(8,8), dpi=200) 

layouts = np.load('rdm_layouts.npy', allow_pickle=True)

EXTENT = 30
for i in range(N_LAYOUTS):
    ax = fig.add_subplot(gs[i])
    ax.set(aspect='equal')
    layout = layouts[i]
    ax.scatter(layout[:,0],layout[:,1],marker='x')
    ax.set(xlim=(-EXTENT,EXTENT),ylim=(-EXTENT,EXTENT))

#%% histogram
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5),dpi=200)
ax.hist(n_turbs)


#%%

min_r  = 2
width = 40
sampler = PoissonDisc(width,width,min_r)
coords = sampler.sample()
layout = np.asarray(coords)

n = len(layout)
pd = (n*np.pi*(min_r/2)**2)/(width+2*min_r)**2
print("packing_density: {}".format(pd))

w = np.sqrt((n*np.pi*min_r**2)/(4*pd))-2*min_r
print("w: {}".format(w))


#%%

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(5,5),dpi=200)
ax.set(aspect='equal')
ax.scatter(layout[:,0],layout[:,1],marker='x')
ax.set(xlim=(0,width),ylim=(0,width))

