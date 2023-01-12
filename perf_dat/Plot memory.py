# plot memory usage from .dat file in same folder
# File format: 1st line: header
# 1st column: useless
# 2nd column: memory usage in MB
# 3rd column: time (from time() function))

import matplotlib.pyplot as plt
import numpy as np

def plot_memory_usage(filename):
    # read data
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    # plot
    plt.plot(data[:, 1])
    plt.xlabel('time (s)')
    plt.ylabel('memory usage (MB)')
    plt.show()

plot_memory_usage('test.dat')
