import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('/codes/auslese/test/oszi_average_20250526_124620.csv', delimiter=',', skiprows=1)


plt.plot(data[:,0], data[:,1])
plt.grid()
#plt.xlim(-0.0002,0.0002)
plt.show()