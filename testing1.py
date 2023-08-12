#%% making sure two version of num_f agree with each other
import matplotlib.pyplot as plt
import numpy as np



theta_i = np.linspace(0,2*np.pi,360,endpoint=False)
U_i = np.ones_like(theta_i)

west = False
if west:
    U_i[180:] = 0
    aU_i = np.append(0,U_i)
    atheta_i = np.append(0,theta_i)
else:
    U_i[:180] = 0
    aU_i = np.append(0,U_i)
    atheta_i = np.append(0,theta_i)
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(atheta_i,aU_i)
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
