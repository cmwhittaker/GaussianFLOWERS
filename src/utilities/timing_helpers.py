#%% This are all most of the AEP functions wrapped in functions that time their execution using %timeit 
# The %command's are interactive python line magic commands, so I think you need Ipython installed to run them
#% times = %timeit -o -q f(x) means: time f(x),(-o) save result as times object, (-q) without feedback. Should take about 7-14 seconds

import numpy as np
import timeit
def adaptive_timeit(func,timed=True):
    #func must take no arguments
    result = func()
    if timed is not True: #don't bother timing
        return result,np.NaN

    number = 5  # 5 iterations to start
    while True:
        # Time how long it takes for 'number' iterations
        elapsed_time = timeit.timeit(lambda: func(), number=number)
        if elapsed_time >= 0.75: 
            break
        number *= 2  # Double number of iterations

    # Now use 'repeat' to run the test multiple times
    times = timeit.repeat(lambda: func(), number=number, repeat=5)
    #this should take ~4-8 secs
    return result,min(times)/number  # Return the best time

from pathlib import Path
from floris.tools import FlorisInterface
def floris_timed_aep(U_i, P_i, theta_i, layout, turb, wake=True, timed=True):
    settings_path = Path("utilities") / "floris_settings.yaml"
    fi = FlorisInterface(settings_path)
    fi.reinitialize(wind_directions=theta_i, wind_speeds=U_i, time_series=True, layout_x=turb.D*layout[:,0], layout_y=turb.D*layout[:,1])

    if wake:
        _,time = adaptive_timeit(fi.calculate_wake,timed=timed)
    else:
        _,time = adaptive_timeit(fi.calculate_no_wake,timed=timed)

    aep_array = fi.get_turbine_powers()
    pow_j = np.sum(P_i[:, None, None]*aep_array, axis=0)  # weight average using probability
    return pow_j/(1*10**6), time
