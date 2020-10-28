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

#kernel = RBF(15 , (1 , 40))
kernel2 = Matern(15 , (1 , 20), nu=1.5)
kernel = Matern(15 , (1 , 20), nu=1.5)
#kernel2 = RBF(15 , (1 , 40))
DIM = 4

x1 = np.random.uniform(XL, XU, (2, DIM))
x2 = np.array(list(x1) + list(np.random.uniform(XL, XU, (1, DIM))))

for __ in range(2000):

   fHs = np.array([delta(np.array([xc]))[0] for xc in x1])
   fLs = np.array([f(np.array([xc]), lf=True)[0] for xc in x2])
   fHs2 = np.array([delta(np.array([xc]))[1] for xc in x1])
   fLs2 = np.array([f(np.array([xc]), lf=True)[1] for xc in x2])
   
   
   gpdelta = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta.fit(np.atleast_2d(x1), fHs)
   #gp1 = GaussianProcessRegressor(kernel=gpdelta.kernel_, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp1 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp1.fit(np.atleast_2d(x2), fLs)
   gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp2.fit(np.atleast_2d(x2), fLs2)
   gpdelta2.fit(np.atleast_2d(x1), fHs2)
   
   #xx = np.array([np.linspace(XL, XU, 100) for _ in range(DIM)])
   #p1, s1 = gp1.predict(np.atleast_2d(xx).T, return_std=True)
   #p2, s2 = gp2.predict(np.atleast_2d(xx).T, return_std=True)
   #pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
   #pd2, sd2 = gpdelta2.predict(np.atleast_2d(xx).T, return_std=True)
   
   from tools import is_pareto_efficient_simple, expected_improvement, KG, parEI, EHI
   
   #EIS = expected_improvement(
   def gpr(x, return_std=False):
      if return_std:
         mu1, s1 = gp1.predict(np.atleast_2d(x), return_std=True)
         mud, sd = gpdelta.predict(np.atleast_2d(x), return_std=True)
         return np.array([mu1 + mud, np.sqrt(s1 ** 2 + sd ** 2)])
      else: 
         mu1 = gp1.predict(np.atleast_2d(x), return_std=False)
         mud = gpdelta.predict(np.atleast_2d(x), return_std=False)
         return mu1 + mud
   
   def gpr1(x, return_std=False):
      if return_std:
         mu1, s1 = gp1.predict(np.atleast_2d(x), return_std=True)
         mud, sd = gpdelta.predict(np.atleast_2d(x), return_std=True)
         return np.array([mu1 + mud, s1])
      else: 
         mu1 = gp1.predict(np.atleast_2d(x), return_std=False)
         mud = gpdelta.predict(np.atleast_2d(x), return_std=False)
         return mu1 + mud

   def gpr2d(x, return_std=False):
      if return_std:
         mu1, s1 = gp2.predict(np.atleast_2d(x), return_std=True)
         mud, sd = gpdelta2.predict(np.atleast_2d(x), return_std=True)
         return np.array([mu1 + mud, np.sqrt(s1 ** 2 + sd ** 2)])
      else: 
         mu1 = gp2.predict(np.atleast_2d(x), return_std=False)
         mud = gpdelta2.predict(np.atleast_2d(x), return_std=False)
         return mu1 + mud
   
   def gpr2(x, return_std=False):
      if return_std:
         mu1, s1 = gp2.predict(np.atleast_2d(x), return_std=True)
         mud, sd = gpdelta2.predict(np.atleast_2d(x), return_std=True)
         return np.array([mu1 + mud, s1])
      else: 
         mu1 = gp2.predict(np.atleast_2d(x), return_std=False)
         mud = gpdelta2.predict(np.atleast_2d(x), return_std=False)
         return mu1 + mud

   #ehi1 = np.array([EHI(xc, gpr1, gpr2, MD=DIM, PCE=True) for xc in xx.T])
   #ehid = np.array([EHI(xc, gpr, gpr2d, MD=DIM, PCE=True) for xc in xx.T])
   eh1min = -99
   ehdmin = -99
   for x0 in [np.random.uniform(XL, XU, DIM) for _ in range(10)]:
        sol = ps(EHI, x0, bounds=[(XL, XU) for _ in range(DIM)], deltaX=3, args={'gp1': gpr, 'gp2': gpr2d, 'MD':DIM, 'NSAMPS':2000, 'PCE':True, 'ORDER': 5})
        if sol['f'] > ehdmin:
           ehdmin = sol['f']
           ehdminx = sol['x']
        sol2 = ps(EHI, x0, bounds=[(XL, XU) for _ in range(DIM)], deltaX=3, args={'gp1': gpr1, 'gp2': gpr2, 'MD':DIM, 'NSAMPS':2000, 'PCE':True, 'ORDER': 5})
        if sol2['f'] > eh1min:
           eh1min = sol2['f']
           eh1minx = sol2['x']

   print("MAX IS ",  np.max([eh1min / 0.05, ehdmin]))
   if np.max([eh1min / 0.05, ehdmin]) < 1e-4: break
   if eh1min / 0.05 > ehdmin:
      x2 = np.append(x2, eh1minx, 0)
   else:
      x1 = np.append(ehdminx, x1, 0)
      x2 = np.append(ehdminx, x2, 0)

fl = open('costLogMD.log', 'w')

fl.write('model evals\n')
fl.write('high %i\n' % x1.size)
fl.write('low %i\n' % x2.size)
fl.write('Best Power: %s (%s)\n' % (str(np.min(pd + p1)), str(xx[:, np.argmin(pd + p1)])))
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
