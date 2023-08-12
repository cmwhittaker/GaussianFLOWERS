#%% These are some useful plotting functions that are shared between most of the plots
import numpy as np
def nice_polar_plot_v01(fig,gs,x,y,ann_txt,bar=True,ylim=None,rlp=0):
    ax = fig.add_subplot(gs,projection='polar')
    if bar:
        ax.bar(x,y,color='grey',linewidth=1,width=2*np.pi/72)
    else:
        ax.plot(x,y,color='black',linewidth=1)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_xticklabels(['N', '', '', '', '', '', '', ''])
    ax.xaxis.set_tick_params(pad=-5)
    ax.set_rlabel_position(rlp)  # Move radial labels away from plotted line
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0.1)
    ax.annotate(ann_txt, xy=(0.4,0.75), ha='center', va='bottom',color='black',xycoords='axes fraction',rotation='vertical',bbox=props)
    ax.spines['polar'].set_visible(False)
    ax.set(ylim=ylim)
    return None