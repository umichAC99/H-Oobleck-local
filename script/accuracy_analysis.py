import matplotlib.pyplot as plt
import numpy as np

# Sample data: Converted from milliseconds to seconds
runtimes_optimal = [100488.554588, 52810.885161, 33138.643266, 31506.181207, 31989.985589]
runtimes_approx = [101468.183774, 54894.842745, 34023.693034, 32673.192131, 34559.451579]

# Convert milliseconds to seconds
runtimes_optimal = [x / 1000 for x in runtimes_optimal]
runtimes_approx = [x / 1000 for x in runtimes_approx]

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
ax.set_ylabel('Iteration Time (s)')
ax.set_title('Iteration time comparison between Ditto and Optimal')
ax.set_xticks(ind)
ax.set_xticklabels([f'Exp {i+1}' for i in range(n)])
ax.legend()

# Label with bar heights
for bar in optimal_bars + approx_bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.show()
plt.savefig('execution_time_comparison.png')
