#%% Figure of the wind rose locations
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

current_script_path = os.path.abspath(__file__)

# Navigate up one level to the GaussianFLOWERS directory
parent_directory = os.path.dirname(current_script_path)  # This takes you to the src directory
gaussian_flowers_directory = os.path.dirname(parent_directory)  

folder_path = os.path.join(gaussian_flowers_directory, 'data', 'WindRoseData_C')

latitudes = []
longitudes = []
site_numbers = []


# Iterate over each file in the directory
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        # Read the specific lines from the CSV file
        with open(file_path, 'r') as file:
            site_number = int(filename.replace('site', '').replace('.csv', ''))
            header_line = file.readline()  # Read the first line where the titles are
            headers = header_line.split(',')  # Split the line into a list of headers
            longitude = float(headers[7])
            latitude = float(headers[9])
            site_numbers.append(site_number)

        
        longitudes.append(float(longitude))
        latitudes.append(float(latitude))

# Set up the plot with basemap
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(3,2.5),dpi=400)
map = Basemap(projection='merc', llcrnrlat=24, urcrnrlat=52, llcrnrlon=-130,urcrnrlon=-60,resolution='i',ax=ax)

# Draw map details
map.drawcoastlines(linewidth=0.5,color='grey')
map.drawcountries(linewidth=0.5,color='black')
map.drawstates(linewidth=0.5,color='grey')
map.shadedrelief(scale=0.25,alpha=1)

# Convert latitude and longitude to map projection coordinates
x, y = map(longitudes, latitudes)

props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none',pad=0)

for i, txt in enumerate(site_numbers):
    map.scatter(x[i], y[i], s=7, marker='x', color='black', zorder=2,lw=0.5)  # Adjust size with 's' parameter

    ax.annotate(str(txt), (x[i], y[i]), textcoords="offset points", xytext=(0,5), ha='center',va='center',color='black',fontsize=6,zorder=10,bbox=props)
# Show the plot
ax.set_frame_on(False)

from pathlib import Path
current_file_path = Path(__file__)
fig_dir = current_file_path.parent.parent / "fig images"
fig_name = f"Fig_site_map.png"
path_plus_name = fig_dir / fig_name

plt.savefig(path_plus_name, dpi='figure', format='png', bbox_inches='tight')

print(f"figure saved as {fig_name}")
print(f"to {path_plus_name}")