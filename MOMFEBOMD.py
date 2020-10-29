import numpy as np
import matplotlib.pyplot as plt
from patternSearch import patternSearch as ps
from scipy.stats import norm
from twofunc import f as turbF, g
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.interpolate import Rbf
from scipy.optimize import minimize as mini
from scipy.optimize import fmin_cobyla
import matplotlib.ticker as ticker
from tools import XL, XU
plt.style.use('dark_background')
np.random.seed(17)

OPTIMIZER = 'fmin_l_bfgs_b'

# discrepency function (assumes no cost to LF model?)
def delta(x):
   return (np.array([turbF(x, lf=False, MD=True), g(x, lf=False)])
          - np.array([turbF(x, lf=True, MD=True), g(x, lf=True)]))

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf, MD=True), g(x, lf=lf)]

kernel = RBF(15 , (1 , 5e2))
DIM = 4

x1 = np.random.uniform(XL, XU, (2, DIM))

for __ in range(2000):

   fHs = np.array([f(np.array([xc]), lf=False)[0] for xc in x1])
   #fLs = np.array([f(np.array([xc]), lf=True)[0] for xc in x2])
   fHs2 = np.array([f(np.array([xc]), lf=False)[1] for xc in x1])
   #fLs2 = np.array([f(np.array([xc]), lf=True)[1] for xc in x2])
   
   
   gpdelta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta.fit(np.atleast_2d(x1), fHs)
   gpdelta2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta2.fit(np.atleast_2d(x1), fHs2)
   
   l = np.linspace(XL, XU, 10)
   xx = np.random.uniform(XL, XU, (4, 400))
   
   from tools import is_pareto_efficient_simple, expected_improvement, KG, parEI, EHI
   
   #EIS = expected_improvement(
   def gpr(x, return_std=False):
      if return_std:
         mud, sd = gpdelta.predict(np.atleast_2d(x), return_std=True)
         return np.array([mud, np.sqrt(sd ** 2)])
      else: 
         mud = gpdelta.predict(np.atleast_2d(x), return_std=False)
         return mud
   
   def gpr2d(x, return_std=False):
      if return_std:
         mud, sd = gpdelta2.predict(np.atleast_2d(x), return_std=True)
         return np.array([mud, np.sqrt(sd ** 2)])
      else: 
         mud = gpdelta2.predict(np.atleast_2d(x), return_std=False)
         return mud
   
   print('probe1')
   ehid = np.array([EHI(xc, gpr, gpr2d, MD=DIM, NSAMPS=50) for xc in xx.T])
   print('prober21')
   #eid = EHI(15, gpr, gpr1)
   #ei1 = -1 * expected_improvement(xx, x2, fHs + fLs[:fHs.size], gpr1, xi=XI)
   #eid = -1 * expected_improvement(xx, x1, fHs + fLs[:fHs.size], gpr, xi=0.01)
   #ax[2, 0].plot(xx, eid, label=r'$EI(\mu_\delta)$', c='r')

   if np.max([ehid]) < 1e-2: break
   x1 = np.append(np.atleast_2d(xx[:, np.argmax(ehid)]), x1, 0)

l = np.linspace(XL, XU, 60)
xx = np.meshgrid(l, l, l, l)[0].reshape(DIM, l.size ** (DIM) // DIM)
pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
pd2, sd2 = gpdelta2.predict(np.atleast_2d(xx).T, return_std=True)

fl = open('BOcostLogMD.log', 'w')

fl.write('model evals\n')
fl.write('high %i\n' % (x1.size // DIM))
xsol = xx[:, np.argmin(pd)]
fl.write('Best Power: %s (%s)\n' % (str(f([xsol], lf=False)), str(xsol)))
fl.close()

