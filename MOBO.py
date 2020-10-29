import numpy as np
from tools import is_pareto_efficient_simple, expected_improvement, KG, parEI, EHI
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
kernel = RBF(15 , (10 , 5e2 ))

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf)[0], g(x, lf=lf)]

# initial samples
x1 = np.random.uniform(XL, XU, 2)

for __ in range(20000):

   # summon model evaluations
   fHs = np.array([f(np.array([xc]), lf=False)[0] for xc in x1])
   fHs2 = np.array([f(np.array([xc]), lf=False)[1] for xc in x1])
   
   # Fit GPs
   gpdelta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta.fit(np.atleast_2d(x1).T, fHs)
   gpdelta2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta2.fit(np.atleast_2d(x1).T, fHs2)
   
   # Make predicitons for plotting
   xx = np.linspace(XL, XU, 100)
   pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
   pd2, sd2 = gpdelta2.predict(np.atleast_2d(xx).T, return_std=True)
   
   # Define helper functions for EHVI computaiton
   def gpr(x, return_std=False):
      if return_std:
         mud, sd = gpdelta.predict(np.atleast_2d(x).T, return_std=True)
         return np.array([mud, np.sqrt(sd ** 2)])
      else: 
         mud = gpdelta.predict(np.atleast_2d(x).T, return_std=False)
         return mud
   
   def gpr2d(x, return_std=False):
      if return_std:
         mud, sd = gpdelta2.predict(np.atleast_2d(x).T, return_std=True)
         return np.array([mud, np.sqrt(sd ** 2)])
      else: 
         mud = gpdelta2.predict(np.atleast_2d(x).T, return_std=False)
         return mud

   # estimate EHVI aross grid
   ehid = np.array([EHI(xc, gpr, gpr2d) for xc in xx])

   # Create plots
   fig, ax = plt.subplots(2, 2, sharex=False, figsize=(10, 5))
   plt.subplots_adjust(wspace=0.2)

   ax12 = ax[0, 0].twinx()
   ax12.fill_between(xx, pd2 - sd2, pd2 + sd2, facecolor='purple', alpha=0.7)

   ax[0, 0].fill_between(xx, pd - sd, pd + sd, facecolor='red', alpha=0.7)
   ax[0, 0].plot(xx, pd, c='red')
   ax[0, 0].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
   ax[0, 0].scatter(x1, fHs, c='w', marker='x')

   ax12.plot(xx, pd2, c='purple')
   ax12.plot(xx, [f(np.array([xc]), lf=False)[1] for xc in xx], c='yellow', ls='--')
   ax12.scatter(x1, fHs2, c='w', marker='x')

   ax[1, 0].set_xlabel('x')
   ax[0, 0].set_ylabel(r'High-Fidelity')
   

   ax[0, 0].set_title(r'$l_{1} = %.2f, l_{2} = %.2f$' % (gpdelta.kernel_.get_params()['length_scale'], gpdelta2.kernel_.get_params()['length_scale']))
   ax[0, 1].set_visible(False)
   a, b, c = parEI(gpr, gpr2d, x1, np.array([fHs, fHs2 ]), EI=False)
   ax[1, 1].scatter(b[:, c].T[:, 0], b[:, c].T[:, 1], c='red')
   a, b, c = parEI(gpr, gpr2d, x1, np.array([fHs, fHs2]), EI=False, truth=True)
   ax[1, 1].scatter(b[:, c].T[:, 0], b[:, c].T[:, 1], c='yellow')
   ax[1, 0].plot(xx, ehid, label='EHVI($\mu_\delta$)', c='red')
   ax[1, 0].legend()
   ax[1, 1].set_xlabel('$f_1$')
   ax[1, 1].set_ylabel('$f_2$')
   plt.savefig('BO_%03d' % __)
   plt.clf()

   # check stopping condition
   if np.max([ehid]) < 1e-4: break

   # add new point
   x1 = np.append(xx[np.argmax(ehid)], x1)

# Log results
fl = open('BOcostLog.log', 'w')
fl.write('model evals\n')
fl.write('high %i\n' % x1.size)
fl.write('Best Power: %s (%s)\n' % (str(np.min(pd)), str(xx[np.argmin(pd)])))
fl.close()

