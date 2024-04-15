import matplotlib.pyplot as plt
import numpy as np

# Sample data: Replace these lists with your actual runtime data
runtimes_optimal = [100488.554588, 52810.885161, 39272.289252]
runtimes_approx = [101468.183774, 54894.842745, 48348.336041]

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
ax.set_ylabel('Template Execution Time (s)')
ax.set_title('Template execution time comparison between Optimal and Approximate Solutions')
ax.set_xticks(ind)
ax.set_xticklabels([f'Opt{i+1}, Approx{i+1}' for i in range(n)])
ax.legend()

# Adding some text for labels, title and custom x-axis tick labels, etc.
plt.show()
plt.savefig('accuracy_comparison.png')