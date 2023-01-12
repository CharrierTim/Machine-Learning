# plot memory usage from .dat file in same folder

import matplotlib.pyplot as plt
import numpy as np

def plot_memory_usage():
    # ignore first line and column
    data = np.loadtxt('mprofile_20230112184759.dat', skiprows=1, usecols=range(1, 2))
    plt.plot(data)
    plt.xlabel('Time (s)')
    plt.ylabel('Memory usage (MB)')
    plt.show()

if __name__ == '__main__':
    plot_memory_usage()
