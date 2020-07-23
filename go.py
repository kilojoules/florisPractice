from func import f
import numpy as np
from scipy.optimize import minimize as mini
outf = open('skippy.log', 'w')
outf.write('y1 y2 y3 y4 pow\n')
def callb(x):
    outf.write(' '.join([str(xx) for xx in x] + [str(f(x)), '\n']))
s = mini(f, np.zeros(4), bounds = [(-30, 30) for _ in range(4)], callback=callb)
outf.close()
