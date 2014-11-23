from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for clr, z in zip(['r', 'g', 'b', 'y', 'c','m','k','r','g','b'], range(21,31)):
    xs = c[(z-21)*24:(z-20)*24]["hour"]
    ys = c[(z-21)*24:(z-20)*24]["CTR"]
    cs = [clr] * len(xs)
    cs[0] = clr
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

