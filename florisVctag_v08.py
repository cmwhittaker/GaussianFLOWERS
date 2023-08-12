#%% Evaluation of FLORIS vs CTAG vs CubedAv (3-way)
# Now with the *new* discrete wake function and the *new* NTAG
import numpy as np
no_xt = 6 #number of turbines in x
no_yt = 4
SITE = 11
def rectangular_layout(s=5):
    xt = np.arange(1,no_xt+1,1)*s
    yt = np.arange(1,no_yt+1,1)*s
    Xt,Yt = np.meshgrid(xt,yt)
    return Xt.reshape(-1),Yt.reshape(-1),np.column_stack((Xt.reshape(-1),Yt.reshape(-1))), np.size(Xt)#just a single layout for now

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
no_bins = 72
from distributions_vC05 import wind_rose
wr = wind_rose(bin_no_bins=no_bins,custom=None,a_0=8,site=SITE,Cp_f=turb.Cp_f)

spacing = 5
xt,yt,layout, Nt = rectangular_layout(s=spacing)
X,Y,plot_points = rectangular_domain(layout,s=spacing,pad=3,r=100)

K = 0.03
from AEP3_2_functions import ntag_v01,cubeAv_v3

from floris.tools import FlorisInterface
fi = FlorisInterface("floris_settings.yaml")

b0,c0 = calc_floris_powers(layout,
                       wr.avMagnitude,
                       wr.frequency,
                       turb.D)

a1,b1,c1 = cubeAv_v3(layout,layout,
                        np.linspace(0,2*np.pi,no_bins,endpoint=False),
                        wr.avMagnitude,
                        wr.frequency,
                        turb.Ct_f,
                        turb.Cp_f,
                        K,
                        turb.A)

a2,b2,c2 = ntag_v01(layout,layout,
                    wr.cjd3_full_Fourier_coeffs,
                    turb.Ct_f(np.sum(wr.frequency*wr.avMagnitude)),
                    K,
                    turb.A)

U_ref,_,_ =  cubeAv_v3(layout,plot_points,
                        np.linspace(0,2*np.pi,no_bins,endpoint=False),
                        wr.avMagnitude,
                        wr.frequency,
                        turb.Ct_f,
                        turb.Cp_f,
                        K,
                        turb.A)

b4,c4 = calc_floris_powers(layout, #reference without wake...
                       wr.avMagnitude,
                       wr.frequency,
                       turb.D,wake=False)

def pce(approx,exact,rel):
    #rel is a modification to make the error more realistic
    return 100*(exact-approx)/(exact-rel)

def nice_polar_plot(fig,gs,x,y,text):
    #first column is the wind rose
    ax = fig.add_subplot(gs,projection='polar')
    ax.plot(x,y,color='black',linewidth=2)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(60)  # Move radial labels away from plotted line
    ax.text(0, 0, text, ha='left',transform=ax.transAxes,color='black')
    ax.spines['polar'].set_visible(False)
    return None

cmtoI = 1/2.54  # centimeters in inches
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 4, height_ratios=[1,0.05,0.3],width_ratios=[0.25,0.25,0.25,0.25],wspace=0.1,hspace=0.3)
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure(figsize=(19*cmtoI,19*cmtoI),dpi=200)
#was 19cm, other was 25
#first is the contourf
ax1 = fig.add_subplot(gs[0,:])
cf = ax1.contourf(X,Y,U_ref.reshape(X.shape),50,cmap=cm.coolwarm)

for j in range(Nt): #for each turbine
    #label that turbine with text of the power output and the percentage difference
    #need to write the sign convention in the report
    label_text = "N:{}\n{:.2f}MW\n{:.2f}({:.2f}\%)\n{:.2f}({:.2f}\%)".format(j+1,b0[j],b1[j],pce(b1[j],b0[j],b4[j]),b4[j],pce(b2[j],b0[j],b4[j]))
    ax1.text(xt[j],yt[j],label_text,fontsize=7,ha='center',va='center')
ax1.scatter(xt,yt,marker='x',color='white')

props = dict(boxstyle='round', facecolor='white', alpha=0.8)
top_left_text = "FARM AEP\nfloris: {:.2f}MW\ncubeAv: {:.2f}({:.2f}\%)MW\nntag: {:.2f}({:.2f}\%)\nNOWAKE: {:.2f}({:.2f}\%)".format(c0,c1,pce(c1,c0,c4),c2,pce(c2,c0,c4),c4,pce(c4,c0,c4))
ax1.text(0.1,0.92,top_left_text,color='black',transform=ax1.transAxes,va='center',ha='center',fontsize=6,bbox=props)

top_right_text = "KEY\nfloris_power / mw \nctag_power(\% error)\ndiscrete numerical power(\% error)"
ax1.text(0.8,0.92,top_right_text,color='black',transform=ax1.transAxes,va='center',ha='center',fontsize=6,bbox=props)

ax1.text(0.02,0.02,"site:{}, k:{}".format(SITE,K),color='black',transform=ax1.transAxes,va='bottom',ha='left',fontsize=10,bbox=props)

#then the colourbar
cax = fig.add_subplot(gs[1,:])
cb = fig.colorbar(cf, cax=cax, orientation='horizontal',format='%.3g')
#then the probability
xs = np.linspace(0,2*np.pi,no_bins,endpoint=False)
nice_polar_plot(fig,gs[2,0],xs,wr.frequency,'$P[\\theta]$')
#then the speed
nice_polar_plot(fig,gs[2,1],xs,wr.avMagnitude,'$U[\\theta]$')
#the joint
nice_polar_plot(fig,gs[2,2],xs,turb.Cp_f(wr.avMagnitude)*wr.avMagnitude*wr.frequency,'$C_p(U[\\theta])P[\\theta]U[\\theta]$')
#then Ct variation
nice_polar_plot(fig,gs[2,3],xs,turb.Ct_f(wr.avMagnitude),'$C_t(U[\\theta])$')

filepath = r"AEP3_Evaluation_Report_v02\Figures\florisVctag_v08_"+str(no_xt) + "x" + str(no_yt) + "_site" +str(SITE) + ".png"
plt.savefig(filepath,dpi='figure',format='png',bbox_inches='tight')

#%% set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)
