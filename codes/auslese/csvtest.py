import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/rojinaksu/PycharmProjects/diy-particle-detector-V2/codes/auslese/oszi_average_20250523_121021.csv', delimiter=',', skiprows=1)


plt.plot(data[:,0], data[:,1])
plt.show()