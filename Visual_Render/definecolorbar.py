import matplotlib.pyplot as plt
import numpy as np

# # Generate some data to plot
# x = np.linspace(0, 10, 100)
# y = x ** 2
# z = x ** 3
#
# # Create the plot
# fig, ax = plt.subplots()
# im = ax.scatter(x, y, c=z, cmap='jet', vmin=0, vmax=0.03)
#
# # Create the colorbar
# cbar = plt.colorbar(im, ax=ax)
#
# # Set the tick locations and labels
# cbar.set_ticks(np.arange(0, 0.035, 0.005))
# cbar.set_ticklabels(np.arange(0, 0.035, 0.005))
# cbar.set_label('Uncertainty',fontsize =15)
#
# # Show the plot
# plt.show()
import matplotlib.pyplot as plt
import matplotlib

# Define the colors for the colorbar
colors = [(1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.502)]

# Create the colorbar
fig, ax = plt.subplots(figsize=(.5,6))
cmap = matplotlib.colors.ListedColormap(colors)
cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, orientation='vertical')

# Set the ticks and labels for the colorbar
cb.set_ticks([0.0, 10.0, 20.0, 30.0, 40.0])
cb.set_ticklabels(['0.0', '10.0', '20.0', '30.0', '40.0'])
cb.set_label('Minimum distance; mm')

# Show the plot
plt.show()