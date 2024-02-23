def set_latex_font():
    #try to set the nice latex font on the latex figures
    from matplotlib import rc
    try:
        rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':9})
        rc('text', usetex=True)
    except:
        print("Could not set matplotlib to use latex font ")
    return None

def si_fm(number):
    # Display a value to 3dp in scientific forma using SI prefix
    # e.g. 1234 is 1.23k

    if number == np.nan:
        return "NAN"

    prefixes = {
        24: 'Y',  # yotta
        21: 'Z',  # zetta
        18: 'E',  # exa
        15: 'P',  # peta
        12: 'T',  # tera
        9: 'G',   # giga
        6: 'M',   # mega
        3: 'k',   # kilo
        0: '',    # (no prefix)
        -3: 'm',  # milli
        -6: 'µ',  # micro
        -9: 'n',  # nano
        -12: 'p', # pico
        -15: 'f', # femto
        -18: 'a', # atto
        -21: 'z', # zepto
        -24: 'y'  # yocto
    }
    # Find the appropriate prefix for the number
    for exp, prefix in prefixes.items():
        if number >= 10 ** exp:
            break
    value = round(number / (10 ** exp), 3)

    # Return the formatted string
    return f"{value}{prefix}"

import numpy as np
def nice_polar_plot_A(fig,gs,x,y,ann_txt,bar=True,ylim=None,rlp=0):
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