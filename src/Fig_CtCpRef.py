#% A figure to show the thrust and power coefficient of the chosen turbine. 

#%% Set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)
import numpy as np
from turbines_v01 import iea_10MW
turb = iea_10MW()
ws = np.linspace(0,27,200)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(3,3),dpi=250)
ax.plot(ws,turb.Ct_f(ws),label='$C_t$',color='black')
ax.plot(ws,turb.Cp_f(ws),label='$C_p$',color='blue')

ax.set_xlabel('Wind Speed / $ms^{-1}$',labelpad=0)
ax.set_ylabel('$C_t$,$C_p$',labelpad=0)

ax.legend()
if True:
    print("figure saved!")
    from pathlib import Path
    path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
    plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches="tight")
#%%

#(A 1 x 3 pannel of the wake velocity deficit error for a SINGLE wind direction
import numpy as np
ks=np.array((0.02,0.04,0.06))
Cts=np.array((0.7,0.8,0.9))
eps = 0.2*np.sqrt((1+np.sqrt(1-Cts))/(2*np.sqrt(1-Cts))) 
x_lims = (np.sqrt(Cts/8)-eps)/ks
x_lims = 5

RHO = 1.225
from turbines_v01 import iea_10MW
turb = iea_10MW()
A = turb.A
Cp_f = turb.Cp_f
U_inf = 10

def exact(r,theta,params,U_inf):
    k,ep,Ct = params
    U_w = U_inf*( 1-((1-np.sqrt(1-(Ct/(8*(k*r*np.cos(theta)+ep)**2))))*np.exp((-(r*np.sin(theta))**2)/(2*(k*r*np.cos(theta)+ep)**2))) )
    return 0.5*A*RHO*turb.Cp_f(U_w)*U_w**3

def approx(r,theta,params,U_inf):
    k,ep,Ct = params
    U_w = U_inf*( 1-((1-np.sqrt(1-(Ct/(8*(k*r+ep)**2))))*np.exp((-(r*theta)**2)/(2*(k*r+ep)**2))) )
    return 0.5*A*RHO*turb.Cp_f(U_w)*U_w**3

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 4, height_ratios=[14,1],wspace=0.3,hspace=0.37)

fig = plt.figure(figsize=(7.8,2.2), dpi=300)

#first row, first column is the arrow
ax = fig.add_subplot(gs[0,0])
ax.set(xlim=(0,10),ylim=(0,10))
ax.arrow(2,5,6,0, head_width=0.5, head_length=0.5, fc='k', ec='k')
text = "Single " + str(U_inf)+"ms incoming\n wind direction\n"
ax.text(5,3.4, text, size=8, ha="center", va="top",bbox=dict(boxstyle="square",ec='w', fc='w',pad=0,mutation_aspect=1.2))
ax.axis('off')

#first row, the next three columns are contourf plots
for i in range(3):

    x = np.array([np.linspace(3,10,500)])
    y = np.array([np.linspace(-1.5,1.5,500)])
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2+Y**2).reshape(-1)
    THETA = np.arctan2(Y,X).reshape(-1)

    params = ks[i],eps[i],Cts[i]
    U_w_exact = exact(R,THETA,params,U_inf).reshape(X.shape)
    U_w_approx = approx(R,THETA,params,U_inf).reshape(X.shape)
    
    pc_error = 100*np.abs((U_w_exact-U_w_approx)/U_w_exact)
    pc_error = np.where(X<x_lims,np.nan,pc_error)

    ax = fig.add_subplot(gs[0,i+1])

    xticks = ax.xaxis.get_major_ticks()
    xticks[1].set_visible(False)
    ax.set_xlabel('$x/d_0$',labelpad=-10)
    ax.set_xlim((0,np.max(x)))

    ax.tick_params(axis='y', which='major', pad=1)

    cf = ax.contourf(X,Y,pc_error,10,cmap=cm.coolwarm)
    cax = fig.add_subplot(gs[1,i+1])
    cb = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cb.ax.locator_params(nbins=5)

    text = '$k=' + str(ks[i]) + '$' + ',$C_T=' + str(Cts[i]) + '$' 

    ax.text(0.97*np.max(x), 0.97*np.max(y), text, size=9, ha="right", va="top",bbox=dict(boxstyle="square",ec='w', fc='w',pad=0))

    ax.scatter(0,0,marker = 'x',color='black')
    if i==0:
        ax.set_ylabel('$y/d_0$',labelpad=-12)
        yticks = ax.yaxis.get_major_ticks()
        yticks[2].set_visible(False)
    
    if i==1:
        cax.set_xlabel('Abs Percentage Error in AEP / \%',labelpad=5)

from pathlib import Path
path_plus_name = "JFM_report_v02/Figures/"+Path(__file__).stem+".png"
plt.savefig(path_plus_name,dpi='figure',format='png',bbox_inches="tight")

