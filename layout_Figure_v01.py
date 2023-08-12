#%% Figure to show the layout used in the evalulation of the different wake evaulation methods.
spacing = 5
import numpy as np
xt = np.arange(1,6+1,1)*spacing
yt = np.arange(1,4+1,1)*spacing
Xt,Yt = np.meshgrid(xt,yt)
Xt,Yt = Xt.reshape(-1),Yt.reshape(-1)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(4,4),dpi=400)
ax.scatter(Xt,Yt,marker='x')
ax.set_aspect('equal')

bbox = fig.bbox_inches.from_bounds(0.1,0.6,3.6,2.5) #Crop
plt.savefig(r"AEP3_Evaluation_Report_v02\Figures\layout_Figure_v01.png",dpi='figure',format='png',bbox_inches=bbox)

#%% set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)