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
COST = 0.01 # LF/HF cost

# discrepency function (assumes no cost to LF model?)
def delta(x):
   return (np.array([turbF(x, lf=False), g(x, lf=False)])
          - np.array([turbF(x, lf=True), g(x, lf=True)]))

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf), g(x, lf=lf)]

kernel = RBF(15 , (1e-2 , 4e2 ))
kernel2 = RBF(15 , (20 , 4e2 ))

# initial samples
x1 = np.random.uniform(XL, XU, 2)
x2 = np.array(list(x1) + list(np.random.uniform(XL, XU, 1)))

# initialize fixed grid
xx = np.linspace(XL, XU, 100)

for __ in range(2000):

   # summon model evaluations
   fHs = np.array([delta(xc)[0] for xc in x1])
   fLs = np.array([f(xc, lf=True)[0] for xc in x2])
   fHs2 = np.array([delta(xc)[1] for xc in x1])
   fLs2 = np.array([f(xc, lf=True)[1] for xc in x2])
   
   # Fit GP models
   gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp1.fit(np.atleast_2d(x2).T, fLs)
   gpdelta = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta.fit(np.atleast_2d(x1).T, fHs)
   gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp2.fit(np.atleast_2d(x2).T, fLs2)
   gpdelta2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta2.fit(np.atleast_2d(x1).T, fHs2)
   
   # Create helper functions for EHI computation
   # (returns predicted HF mean, LF or HF variance)
   def gpr(x, return_std=False):
      if return_std:
         mu1, s1 = gp1.predict(np.atleast_2d(x).T, return_std=True)
         mud, sd = gpdelta.predict(np.atleast_2d(x).T, return_std=True)
         return np.array([mu1 + mud, np.sqrt(s1 ** 2 + sd ** 2)])
      else: 
         mu1 = gp1.predict(np.atleast_2d(x).T, return_std=False)
         mud = gpdelta.predict(np.atleast_2d(x).T, return_std=False)
         return mu1 + mud
   
   def gpr1(x, return_std=False):
      if return_std:
         mu1, s1 = gp1.predict(np.atleast_2d(x).T, return_std=True)
         mud, sd = gpdelta.predict(np.atleast_2d(x).T, return_std=True)
         return np.array([mu1 + mud, s1])
      else: 
         mu1 = gp1.predict(np.atleast_2d(x).T, return_std=False)
         mud = gpdelta.predict(np.atleast_2d(x).T, return_std=False)
         return mu1 + mud

   def gpr2d(x, return_std=False):
      if return_std:
         mu1, s1 = gp2.predict(np.atleast_2d(x).T, return_std=True)
         mud, sd = gpdelta2.predict(np.atleast_2d(x).T, return_std=True)
         return np.array([mu1 + mud, np.sqrt(s1 ** 2 + sd ** 2)])
      else: 
         mu1 = gp2.predict(np.atleast_2d(x).T, return_std=False)
         mud = gpdelta2.predict(np.atleast_2d(x).T, return_std=False)
         return mu1 + mud
   
   def gpr2(x, return_std=False):
      if return_std:
         mu1, s1 = gp2.predict(np.atleast_2d(x).T, return_std=True)
         mud, sd = gpdelta2.predict(np.atleast_2d(x).T, return_std=True)
         return np.array([mu1 + mud, s1])
      else: 
         mu1 = gp2.predict(np.atleast_2d(x).T, return_std=False)
         mud = gpdelta2.predict(np.atleast_2d(x).T, return_std=False)
         return mu1 + mud

   # compute EHVI
   ehi1 = np.array([EHI(xc, gpr1, gpr2, PCE=False) for xc in xx])
   ehid = np.array([EHI(xc, gpr, gpr2d, PCE=False) for xc in xx])

   # make prediction for plotting
   p1, s1 = gp1.predict(np.atleast_2d(xx).T, return_std=True)
   p2, s2 = gp2.predict(np.atleast_2d(xx).T, return_std=True)
   pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
   pd2, sd2 = gpdelta2.predict(np.atleast_2d(xx).T, return_std=True)
   
   # Create plots
   fig, ax = plt.subplots(3, 2, sharex=False, figsize=(10, 5))
   plt.subplots_adjust(wspace=0.2, hspace=0.4)
   ax[0, 0].plot(xx, p1, c='lightblue')
   ax[0, 0].fill_between(xx, p1 - s1, p1 + s1, facecolor='lightblue', alpha=0.7)
   ax02 = ax[0, 0].twinx()
   ax02.plot(xx, p2, c='lightgreen')
   ax02.fill_between(xx, p2 - s2, p2 + s2, facecolor='lightgreen', alpha=0.7)

   ax[0, 0].plot(xx, [f(np.array([xc]), lf=True)[0] for xc in xx], c='yellow', ls='-')
   ax02.plot(xx, [f(np.array([xc]), lf=True)[1] for xc in xx], c='yellow', ls='--')

   ax[0, 0].scatter(x2, fLs, c='w', marker='x')
   ax02.scatter(x2, fLs2, c='w', marker='x')

   ax12 = ax[1, 0].twinx()
   ax12.fill_between(xx, p2 + pd2 - s2 - sd2, p2 + pd2 + s2 + sd2, facecolor='purple', alpha=0.7)
   ax12.fill_between(xx, p2 + pd2 - s2, p2 + pd2 + s2, facecolor='lightblue', alpha=0.7)

   ax[1, 0].fill_between(xx, p1 + pd - s1 - sd, p1 + pd + s1 + sd, facecolor='red', alpha=0.7)
   ax[1, 0].fill_between(xx, p1 + pd - s1, p1 + pd + s1, facecolor='lightblue', alpha=0.7)
   ax[1, 0].plot(xx, p1 + pd, c='red')
   ax[1, 0].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
   ax[1, 0].scatter(x1, fLs[:x1.size] + fHs, c='w', marker='x')

   ax12.plot(xx, p2 + pd2, c='purple')
   ax12.plot(xx, [f(np.array([xc]), lf=False)[1] for xc in xx], c='yellow', ls='--')
   ax12.scatter(x1, fLs2[:x1.size] + fHs2, c='w', marker='x')

   ax[2, 0].set_xlabel('x')
   ax[0, 0].set_ylabel(r'Low-Fidelity')
   ax[1, 0].set_ylabel(r'High-Fidelity')
   
   
   ax[0, 1].set_visible(False)
   ax[1, 1].set_visible(False)
   a, b, c = parEI(gpr, gpr2d, x1, np.array([fHs + fLs[:fHs.size], fHs2 + fLs2[:fHs.size]]), EI=False)
   ax[2, 1].scatter(b[:, c].T[:, 0], b[:, c].T[:, 1], c='red')
   a, b, c = parEI(gpr2, gpr2d, x1, np.array([fHs + fLs[:fHs.size], fHs2 + fLs2[:fHs.size]]), EI=False, truth=True)
   ax[2, 1].scatter(b[:, c].T[:, 0], b[:, c].T[:, 1], c='yellow')
   ax[0, 0].set_title(r'$l_1 = %.2f, l_2 = %.2f$' % (gp1.kernel_.get_params()['length_scale'], gp2.kernel_.get_params()['length_scale']))
   ax[1, 0].set_title(r'$l_{\delta_1} = %.2f, l_{\delta_2} = %.2f$' % (gpdelta.kernel_.get_params()['length_scale'], gpdelta2.kernel_.get_params()['length_scale']))
   ax[2, 0].plot(xx, ehi1 / COST, label='EHVI$(\mu_1) / %f$' % COST, c='lightblue')
   ax[2, 0].plot(xx, ehid, label='EHVI($\mu_\delta$)', c='red')
   ax[2, 0].legend()
   ax[2, 1].set_xlabel('$f_1$')
   ax[2, 1].set_ylabel('$f_2$')
   plt.savefig('MF_%03d' % __)
   plt.clf()
 
   # Check stopping condition
   if np.max(ehid) < 1e-4: break

   # Determine which model to sample next
   if np.max(ehi1) / COST > np.max(ehid):
      x2 = np.append(x2, xx[np.argmax(ehi1)])
   else:
      x1 = np.append(xx[np.argmax(ehid)], x1)
      x2 = np.append(xx[np.argmax(ehid)], x2)

# log results
fl = open('costLog.log', 'w')
fl.write('model evals\n')
fl.write('high %i\n' % x1.size)
fl.write('low %i\n' % x2.size)
fl.write('Best Power: %s (%s)\n' % (str(np.min(pd + p1)), str(xx[np.argmin(pd + p1)])))
fl.write('Best Load: %s (%s)\n' % (str(np.min(pd2 + p2)), str(xx[np.argmin(pd2 + p2)])))
fl.close()

