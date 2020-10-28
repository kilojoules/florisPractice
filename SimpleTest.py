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

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf)[0], g(x, lf=lf)]

#kernel = Matern(15 , (4 , 5e2 ), nu=1.5)
kernel = RBF(15 , (8 , 5e2 ))

x1 = np.random.uniform(XL, XU, 2)

for __ in range(200):

   fHs = np.array([f(np.array([xc]), lf=False)[0] for xc in x1])
   fHs2 = np.array([f(np.array([xc]), lf=False)[1] for xc in x1])
   
   
   gpdelta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta.fit(np.atleast_2d(x1).T, fHs)
   gpdelta2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta2.fit(np.atleast_2d(x1).T, fHs2)
   
   xx = np.linspace(XL, XU, 100)
   pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
   pd2, sd2 = gpdelta2.predict(np.atleast_2d(xx).T, return_std=True)
   
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
   #ax[0].set_ylabel(r'$f^1$ (low-fidelity)')
   #ax[1].set_ylabel(r'$f^0$ (high-fidelity)')
   ax[0, 0].set_ylabel(r'High-Fidelity')
   
   from tools import is_pareto_efficient_simple, expected_improvement, KG, parEI, EHI
   
   #EIS = expected_improvement(
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

   #ax[0].fill_between(xx, p1 + pd - s1 - sd, p1 + pd + s1 + sd, facecolor='red', alpha=0.7)
   #ax[0].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
   #ax[0].plot(xx, p1 + pd, c='red')
   a1, sa = gpr(xx, return_std=True)
   #ax[2].fill_between(xx, a1 - sa, a1 + sa, facecolor='red', alpha=0.7)
   #ax[2].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
   #ax[2].plot(xx, a1, c='red')
   #xis = [0, 0.01, 0.02, 0.03, 0.05]
   #colors = plt.cm.coolwarm(np.linspace(0, 1, len(xis)))
   #for kk, xi in enumerate(xis):
#   ax[1].plot(xx, -1 * expected_improvement(xx, x1, fLs[:x1.size] + fHs, gpr, xi=xi), label=r'$\xi=%.2f$' % xi, c=colors[kk])
#ax[1].twinx().plot(xx, -1 * expected_improvement(xx, x1, fHs + fLs[:fHs.size], gpr), label=r'$EI(\mu)$')
   XI = 0.04
   ax[0, 1].set_visible(False)
   a, b, c = parEI(gpr, gpr2d, x1, np.array([fHs, fHs2 ]), EI=False)
   ax[1, 1].scatter(b[:, c].T[:, 0], b[:, c].T[:, 1], c='red')
   a, b, c = parEI(gpr, gpr2d, x1, np.array([fHs, fHs2]), EI=False, truth=True)
   ax[1, 1].scatter(b[:, c].T[:, 0], b[:, c].T[:, 1], c='yellow')
   ehidMC200 = np.array([EHI(xc, gpr, gpr2d, NSAMPS=200) for xc in xx])
   ehidMC1000 = np.array([EHI(xc, gpr, gpr2d, NSAMPS=1000) for xc in xx])
   ehidMC5000 = np.array([EHI(xc, gpr, gpr2d, NSAMPS=5000) for xc in xx])
   #ehiPCE1 = np.array([EHI(xc, gpr, gpr2d, PCE=True, NSAMPS=20, ORDER=1) for xc in xx])
   #ehiPCE2 = np.array([EHI(xc, gpr, gpr2d, PCE=True, NSAMPS=20, ORDER=2) for xc in xx])
   #ehiPCE3 = np.array([EHI(xc, gpr, gpr2d, PCE=True, NSAMPS=20, ORDER=3) for xc in xx])
   #ehiPCE4 = np.array([EHI(xc, gpr, gpr2d, PCE=True, NSAMPS=20, ORDER=4) for xc in xx])
   #ehiPCE5 = np.array([EHI(xc, gpr, gpr2d, PCE=True, NSAMPS=20, ORDER=5) for xc in xx])
   #ehiPCE7 = np.array([EHI(xc, gpr, gpr2d, PCE=True, NSAMPS=2000, ORDER=7) for xc in xx])
   #eid = EHI(15, gpr, gpr1)
   #ei1 = -1 * expected_improvement(xx, x2, fHs + fLs[:fHs.size], gpr1, xi=XI)
   #eid = -1 * expected_improvement(xx, x1, fHs + fLs[:fHs.size], gpr, xi=0.01)
   #ax[2, 0].plot(xx, eid, label=r'$EI(\mu_\delta)$', c='r')
   ax[1, 0].plot(xx, ehidMC200, label='MC, 200 samples')
   ax[1, 0].plot(xx, ehidMC1000, label='MC, 1000 samples')
   ax[1, 0].plot(xx, ehidMC5000, label='MC, 5000 samples')
   #ax[1, 0].plot(xx, ehiPCE1, label='PCE, P=1')
   #ax[1, 0].plot(xx, ehiPCE2, label='PCE, P=2')
   #ax[1, 0].plot(xx, ehiPCE3, label='PCE, P=3')
   #ax[1, 0].plot(xx, ehiPCE4, label='PCE, P=4')
   #ax[1, 0].plot(xx, ehiPCE5, label='PCE, P=5')
   #ax[1, 0].plot(xx, ehiPCE7, label='PCE, P=7')
   ax[1, 0].set_ylabel('EHVI')
   #ax[2, 0].plot(xx, ei1 / .05, label=r'$EI(\mu_1) / 0.05$', c='lightblue')
   ax[1, 0].legend()
   ax[1, 1].set_xlabel('$f_1$')
   ax[1, 1].set_ylabel('$f_2$')
   #ax[1].set_yscale('log')
   plt.savefig('hey')
   plt.clf()
   hey

   if np.max([ehid]) < 1e-4: break
   x1 = np.append(xx[np.argmax(ehid)], x1)

fl = open('BOcostLog.log', 'w')

fl.write('model evals\n')
fl.write('high %i\n' % x1.size)

fl.write('Best Power: %s (%s)\n' % (str(np.min(pd)), str(xx[np.argmin(pd)])))
fl.close()

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
