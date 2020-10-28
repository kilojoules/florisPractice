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
np.random.seed(7)

OPTIMIZER = None#'fmin_l_bfgs_b'

# discrepency function (assumes no cost to LF model?)
def delta(x):
   return (np.array([turbF(x, lf=False)[0], g(x, lf=False)])
          - np.array([turbF(x, lf=True)[0], g(x, lf=True)]))

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf)[0], g(x, lf=lf)]

kernel = RBF(15 , (1 , 5e2 ))

x1 = np.random.uniform(XL, XU, 2)

for __ in range(10):

   fHs = np.array([f(np.array([xc]), lf=False)[0] for xc in x1])
   
   
   gpdelta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   #gpdelta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta.fit(np.atleast_2d(x1).T, fHs)
   
   xx = np.linspace(XL, XU, 500)
   pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
   
   fig, ax = plt.subplots(2, sharex=True)
   ax[0].fill_between(xx, pd - sd, pd + sd, facecolor='red', alpha=0.7)
   ax[0].plot(xx, pd, c='red')
   ax[0].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
   ax[0].scatter(x1, fHs, c='w', marker='x')
   ax[1].set_xlabel('x')
   ax[0].set_ylabel(r'$f^0$')
   
   from tools import is_pareto_efficient_simple, expected_improvement, KG
   
   #EIS = expected_improvement(
   def gpr(x, return_std=False):
      if return_std:
         mud, sd = gpdelta.predict(np.atleast_2d(x).T, return_std=True)
         return np.array([mud, sd])
      else: 
         mud = gpdelta.predict(np.atleast_2d(x).T, return_std=False)
         return mud
   
   a1, sa = gpr(xx, return_std=True)
   XI = 0.0
   #ei1 = -1 * expected_improvement(xx, x2, fHs + fLs[:fHs.size], gpr1, xi=XI)
   eid = -1 * expected_improvement(xx, x1, fHs, gpr, xi=XI)
   ax[1].plot(xx, eid, label=r'$EI(\mu_\delta)$')
   ax[1].legend()
   ax[0].set_title(r'$\xi=%.2f$' % XI)
   #ax[1].set_yscale('log')
   plt.savefig('SFEGO%i' % __)
   plt.clf()

   x1 = np.append(xx[np.argmax(eid)], x1)

hey
ax2 = ax[1].twinx()
degs = [1, 2, 3, 4, 5]
NSAMPS = 20
PC = True
if PC:
   colors = plt.cm.coolwarm(np.linspace(0, 1, len(degs)))
   for dd, deg in enumerate(degs):
       ax2.plot(xx, [KG(xc, fLs, x2, gpr, kernel, DEG=deg, NSAMPS=NSAMPS) for xc in xx], label='degree = %i' % deg, c=colors[dd])
   ax[0].set_title('%i Samples' % NSAMPS)
else:
   for samp in [5, 10, 50, 100, 500]:
       ax2.plot(xx, [KG(xc, fLs, x2, gpr, kernel, NSAMPS=samp, sampling=True) for xc in xx], label='samples = %i' % samp)
   ax[1].twinx().plot(xx, [KG(xc, fLs, x2, gpr, kernel) for xc in xx], ls='--')
ax2.legend(prop={'size':5})
ax[0].scatter(x2, fLs, c='w', marker='x')
ax[0].set_ylabel('f(x)')
ax[1].set_ylabel('EI(x)')
ax[1].set_xlabel('x')
if PC:
   plt.savefig('KGPCConv2')
else:
   plt.savefig('KGMCConv')
plt.clf()
