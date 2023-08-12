#% A figure to show the thrust and power coefficient of the chosen turbine. 

#%% Set font
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
rc('text', usetex=True)
import numpy as np
from utilities.turbines import iea_10MW
turb = iea_10MW()
ws = np.linspace(0,27,200)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(3,3),dpi=250)
ax.plot(ws,turb.Ct_f(ws),label='$C_t$',color='black')
ax.plot(ws,turb.Cp_f(ws),label='$C_p$',color='blue')

ax.set_xlabel('Wind Speed / $ms^{-1}$',labelpad=0)
ax.set_ylabel('$C_t$,$C_p$',labelpad=0)

ax.legend()

SAVE_FIG = True

if SAVE_FIG:
    from pathlib import Path

    current_file_path = Path(__file__)
    fig_dir = current_file_path.parent.parent / "fig images"
    fig_name = f"Fig_CtCpRef.png"
    image_path = fig_dir / fig_name

    plt.savefig(image_path, dpi='figure', format='png', bbox_inches='tight')

    print(f"figure saved as {fig_name}")