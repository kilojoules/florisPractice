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
plt.style.use('dark_background')
np.random.seed(213)

OPTIMIZER = None #'fmin_l_bfgs_b'

# discrepency function (assumes no cost to LF model?)
def delta(x):
   return (np.array([turbF(x, lf=False)[0], g(x, lf=False)])
          - np.array([turbF(x, lf=True)[0], g(x, lf=True)]))

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf)[0], g(x, lf=lf)]

kernel = RBF(7 , (.3 , 5e2 ))

x1 = np.random.uniform(0, 30, 2)
x2 = np.array(list(x1) + list(np.random.uniform(0, 30, 2)))

fHs = np.array([delta(np.array([xc]))[0] for xc in x1])
fLs = np.array([f(np.array([xc]), lf=True)[0] for xc in x2])


gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
gpdelta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
#gpdelta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
gp1.fit(np.atleast_2d(x2).T, fLs)
gpdelta.fit(np.atleast_2d(x1).T, fHs)

xx = np.linspace(0, 30, 50)
p1, s1 = gp1.predict(np.atleast_2d(xx).T, return_std=True)
pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
stotal = np.sqrt(s1 **2 + sd **2)
s1 *= 2
stotal *= 2

if True:
   fig, ax = plt.subplots(2, sharex=True)
   ax[0].fill_between(xx, p1 - s1, p1 + s1, facecolor='lightblue', alpha=0.7)
   ax[1].fill_between(xx, p1 + pd - stotal, p1 + pd + stotal, facecolor='red', alpha=0.7)
   #ax[1].fill_between(xx, p1 + pd - s1 - sd, p1 + pd + s1 + sd, facecolor='red', alpha=0.7)
   ax[1].fill_between(xx, p1 + pd - s1, p1 + pd + s1, facecolor='lightblue', alpha=0.7)
   ax[0].plot(xx, p1, c='lightblue')
   ax[1].plot(xx, p1 + pd, c='red')
   ax[1].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
   ax[0].plot(xx, [f(np.array([xc]), lf=True)[0] for xc in xx], c='yellow', ls='--')
   ax[0].scatter(x2, fLs, c='w', marker='x')
   ax[1].scatter(x1, fLs[:x1.size] + fHs, c='w', marker='x')
   ax[1].set_xlabel('x')
   ax[0].set_ylabel(r'$f^1$ (low-fidelity)')
   ax[1].set_ylabel(r'$f^0$ (high-fidelity)')
   plt.savefig('teleEx')
   plt.clf()

from tools import is_pareto_efficient_simple, expected_improvement, KG

#EIS = expected_improvement(
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
def gprd(x, return_std=False):
   if return_std:
      mu1, s1 = gp1.predict(np.atleast_2d(x).T, return_std=True)
      mud, sd = gpdelta.predict(np.atleast_2d(x).T, return_std=True)
      return np.array([mu1 + mud, sd])
   else: 
      mu1 = gp1.predict(np.atleast_2d(x).T, return_std=False)
      mud = gpdelta.predict(np.atleast_2d(x).T, return_std=False)
      return mu1 + mud

fig, ax = plt.subplots(2, sharex=True)
#ax[0].fill_between(xx, p1 + pd - s1 - sd, p1 + pd + s1 + sd, facecolor='red', alpha=0.7)
#ax[0].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
#ax[0].plot(xx, p1 + pd, c='red')
a1, sa = gpr(xx, return_std=True)
ax[0].fill_between(xx, a1 - sa, a1 + sa, facecolor='red', alpha=0.7)
#ax[0].fill_between(xx, p1 - s1, p1 + s1, facecolor='red', alpha=0.7)
ax[0].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
ax[0].plot(xx, a1, c='red')
#xis = [0, 0.01, 0.02, 0.03, 0.05]
#colors = plt.cm.coolwarm(np.linspace(0, 1, len(xis)))
#for kk, xi in enumerate(xis):
#   ax[1].plot(xx, -1 * expected_improvement(xx, x1, fLs[:x1.size] + fHs, gpr, xi=xi), label=r'$\xi=%.2f$' % xi, c=colors[kk])
ax[1].plot(xx, -1 * expected_improvement(xx, x1, fHs + fLs[:fHs.size], gpr), label=r'$EI(\mu)$')
ax[1].plot(xx, -1 * expected_improvement(xx, x2, fHs + fLs[:fHs.size], gprd), label=r'$EI(\mu_\delta)$', ls='--')
ax[1].plot(xx, -1 * expected_improvement(xx, x2, fHs + fLs[:fHs.size], gpr1), label=r'$EI(\mu_1)$')
ax[1].legend()
#ax[1].set_yscale('log')
plt.savefig('TelescopingExample')
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
