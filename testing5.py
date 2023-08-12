#%% 
import numpy as np
from turbines_v01 import iea_10MW
turb = iea_10MW()
xs = np.linspace(0,30,100)
yt1 = turb.Ct_f(xs)
yt2 = turb.Cp_f(xs)

import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame(data=[xs,yt1,yt2]).T

filepath = 'my_excel_file.xlsx'

df.to_excel(filepath, index=False)

#%%
import time
ROWS = 2
COLS = 2
LAYS = 3
for i in range(ROWS):
    for j in range(COLS):
        for k in range(LAYS):
            print(f"{(k+1)+j*LAYS+i*LAYS*COLS}/{ROWS*COLS*LAYS}")