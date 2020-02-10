import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import csv

filename = "/tnt/home/kluger/tmp/hlw_analysis.csv"
# filename = "/tnt/home/kluger/tmp/kitti_analysis_5.csv"

cm = plt.cm.get_cmap('plasma')

data = np.genfromtxt(filename, delimiter=' ')

offsets = data[:,0]
angles = data[:,1]

print("offsets: %.6f (%.6f)" % (np.mean(offsets), np.std(offsets)))
print("angles: %.6f (%.6f)" % (np.mean(angles), np.std(angles)))

# matplotlib.rc('xtick', labelsize=26, width=1)
# matplotlib.rc('ytick', labelsize=26)
plt.figure(figsize=(8,8))
ax = plt.gca()
ax.tick_params(labelsize=26, width=4, length=10)
plt.hist2d(data[:,1], data[:,0], bins=32, range=[[-0.05, 0.05], [-0.15, 0.4]], cmap=cm)
plt.show()

