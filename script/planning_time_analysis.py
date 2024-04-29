import matplotlib.pyplot as plt
import numpy as np
import math

# Sample data: Replace these lists with your actual runtime data
runtimes_optimal = [25.505810260772705, 599.8481931686401, 39104.35719227791, 630.139664888382, 7084.061190366745]
runtimes_approx = [2.6304068565368652, 86.63370537757874, 559.2081208229065, 121.43609380722046, 546.0477550029755]

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
ax.set_ylabel('Planning Latency (s)')
ax.set_title('Planning latency comparison between Ditto and Optimal')
ax.set_xticks(ind)
ax.set_xticklabels([f'Exp {i+1}' for i in range(n)])
ax.legend()

# Label with bar heights formatted as minutes or hours
def format_time(seconds):
    if seconds >= 3600:
        return f'{math.ceil(seconds / 3600)}h'  # Hours
    else:
        return f'{math.ceil(seconds / 60)}m'  # Minutes

for bar in optimal_bars + approx_bars:
    height = bar.get_height()
    time_label = format_time(height)
    ax.annotate(time_label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.show()
plt.savefig('planning_time_comparison.png')
