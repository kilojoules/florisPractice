import pandas as pd
from scipy.interpolate import interp1d 
dat = pd.read_csv('./td.csv', sep=', ')
ddat = pd.read_csv('./tdd.csv', sep=', ')  
f = interp1d(dat.x, -1 * dat.y)
