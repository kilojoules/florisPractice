import matplotlib.pyplot as plt 
import pandas as pd
dat = pd.read_csv('./log.log', sep=' ', index_col=False)
sdat = pd.read_csv('./skippy.log', sep=' ', index_col=False)
plt.plot(dat.index, -1 * dat['pow'], c='blue')
plt.plot(sdat.index, -1 * sdat['pow'], c='red')
plt.savefig('hey')
