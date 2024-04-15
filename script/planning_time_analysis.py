import matplotlib.pyplot as plt
import numpy as np

# Sample data: Replace these lists with your actual runtime data
runtimes_optimal = [25.505810260772705, 599.8481931686401, 6676.652161359787]
runtimes_approx = [2.6304068565368652, 86.63370537757874, 270.3531458377838]

# Number of experiments
n = len(runtimes_optimal)

# Creating a figure and a set of subplots
fig, ax = plt.subplots(layout="constrained")

# Index for groups
ind = np.arange(n)  

# The width of the bars
width = 0.35       

# Plotting data
optimal_bars = ax.bar(ind - width/2, runtimes_optimal, width, label='Optimal')
approx_bars = ax.bar(ind + width/2, runtimes_approx, width, label='Approximate')

# Adding labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Experiments')
ax.set_ylabel('Planning Time (s)', font_size=10)
ax.set_title('Planning time comparison between Optimal and Approximate Solutions')
ax.set_xticks(ind)
ax.set_xticklabels([f'Opt{i+1}, Approx{i+1}' for i in range(n)])
ax.legend(font_size=10)

# Adjust layout to prevent clipping of ylabel
plt.tight_layout()

# Adding some text for labels, title and custom x-axis tick labels, etc.
plt.show()
plt.savefig('planning_time_comparison.png')
