import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/rojinaksu/PycharmProjects/diy-particle-detector-V2/codes/Signal tests/Tek000neu.csv',delimiter=',')

x = data[:,0]
y = data[:,1]

plt.plot(x,y, color = 'mediumvioletred')
plt.xlabel('Zeit (s)')
plt.ylabel('Spannung (mV)')
plt.grid()
plt.show()
