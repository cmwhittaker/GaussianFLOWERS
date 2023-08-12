#%% Illustration of the Poisson disc sampling for the report
import numpy as np
from AEP3_functions_v01 import poisson2dRandomPoints
np.random.seed(12341234)
sl = 25
min = 4.9
layouts,distances = poisson2dRandomPoints(1,sl,sl,min_spacing=min,k=30)
layout = layouts[0,:,:]
layout = layout[~np.isnan(layout).any(axis=1)]
print(layout.shape)
print(np.mean(distances))
xt,yt,layout,Nt = layout[:,0],layout[:,1],layout,layout[:,0].size

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10),dpi=400)
ax.scatter(xt,yt)
an_nearest = distances[0]
plt.show()
print("an_nearest: {}".format(an_nearest))
print(f"sl/min: {sl} / {min}")
#%%
import numpy as np

arr = np.array([[1, 2],
                [np.nan, np.nan],
                [4, 5],
                [3, 6]])

# Find the indices of rows with NaN values
nan_rows = np.isnan(arr).any(axis=1)

# Remove rows with NaN values
arr_without_nan = arr[~nan_rows]

print(arr_without_nan)

#%%

#Set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)

from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3,wspace=0.2,hspace=0.2)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,6),dpi=300)

for i in range(9):
    ax = fig.add_subplot(gs[i])
    ax.scatter(layouts[i,:,0],layouts[i,:,1],marker='x',s=14)

bbox = fig.bbox_inches.from_bounds(0,0,5.8,5.8) #Crop
plt.savefig(r"AEP3_Evaluation_Report_v01\Figures\Poisson_disc_illustration.png",dpi='figure',format='png',bbox_inches=bbox)


