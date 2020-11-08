import numpy as np
import itertools
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
from tools import is_pareto_efficient_simple, expected_improvement, KG, parEI, EHI
plt.style.use('dark_background')
np.random.seed(17)
OPTIMIZER = 'fmin_l_bfgs_b'

# discrepency function 
def delta(x):
   return (np.array([turbF(x, lf=False, MD=True), g(x, lf=False)])
          - np.array([turbF(x, lf=True, MD=True), g(x, lf=True)]))

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf, MD=True), g(x, lf=lf)]

def abc(x):
  c = x[0]
  b = x[1:DIM + 1]
  A = np.resize(x[DIM + 1 : DIM * (DIM + 1) + 1], (DIM, DIM))
  return A, b, c

def fitness(x, xin, evls):
   A, b, c = abc(x)
   est = np.einsum("ik,ij,jk->k", xin.T, A, xin.T) + xin.dot(b) + c
   return np.sqrt(np.sum((evls - est) ** 2))

def fitevl(xin, x):
   xin = np.atleast_2d(xin)
   A, b, c = abc(x)
   est = np.einsum("ik,ij,jk->k", xin.T, A, xin.T) + xin.dot(b) + c
   return -1 * est

DELTA_LENGTH_LOW_BOUNDS = 30
kernel = RBF(15 , (1e-2 , 200)) # LF kernel
kernel2 = RBF(15 , (DELTA_LENGTH_LOW_BOUNDS, 200)) # discrepency kernel
kernel3 = RBF(15 , (1e-2 , 200)) # LF kernel
kernel4 = RBF(15 , (DELTA_LENGTH_LOW_BOUNDS, 200)) # LF kernel
DIM = 4

x1 = np.random.uniform(XL, XU, (2, DIM)) # HF samples
x2 = np.array(list(x1) + list(np.random.uniform(XL, XU, (1, DIM)))) # LF samples

# begin optimization
for __ in range(2000):

   x1 = np.round(x1, 5)
   x2 = np.round(x2, 5)

   # summon model evaluations
   fHs = np.array([delta(np.array([xc]))[0] for xc in x1])
   fLs = np.array([f(np.array([xc]), lf=True)[0] for xc in x2])
   fHs2 = np.array([delta(np.array([xc]))[1] for xc in x1])
   fLs2 = np.array([f(np.array([xc]), lf=True)[1] for xc in x2])
   
   # Fit GPs
   gpdelta = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta.fit(np.atleast_2d(x1), fHs2)
   gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp1.fit(np.atleast_2d(x2), fLs2)
   gp2 = GaussianProcessRegressor(kernel=kernel3, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta2 = GaussianProcessRegressor(kernel=kernel4, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   #gp2.fit(np.atleast_2d(x2), fLs2)
   #gpdelta2.fit(np.atleast_2d(x1), fHs2)
   gp2.fit(np.atleast_2d(x2), fLs2)
   gpdelta2.fit(np.atleast_2d(x1), fHs2)
   
   # create mesh
   #l = np.linspace(XL, XU, 10)
   #xx = np.array([xc for xc in itertools.permutations(l, 4)]).T
   #xx = np.meshgrid(l, l, l, l)[0].reshape(DIM, l.size ** (DIM) // DIM)
   #np.random.seed(12)
   xx = np.append(np.random.uniform(XL, XU, (4, 400)), x2.T, 1)
   #xx = np.array([np.linspace(XL, XU, 10) for _ in range(DIM)])
   
   # define helper functions for computing Expected Hypervolume Improvement
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

   # compute EHVI for each point in grid
   ehi1 = np.array([EHI(xc, gpr1, gpr2, MD=DIM, NSAMPS=500) for xc in xx.T])
   ehid = np.array([EHI(xc, gpr, gpr2d, MD=DIM, NSAMPS=500) for xc in xx.T])

   # Quadratic model
   #x0 = np.ones(1 + DIM * (DIM + 1))
   #sol = mini(fitness, x0, args=(xx.T, ehi1))
   #sol2 = mini(fitness, x0, args=(xx.T, ehid))
   #ehsol = mini(fitevl, np.ones(DIM), args=sol.x, bounds=[[XL, XU] for ___ in range(DIM)])
   #ehsol2 = mini(fitevl, np.ones(DIM), args=sol2.x, bounds=[[XL, XU] for ___ in range(DIM)])
   gpeh1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpeh2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpeh2.fit(xx.T, ehid)
   gpeh1.fit(xx.T, ehi1)
   x = np.linspace(XL, XU, 31)
   xxx = np.stack(np.meshgrid(*[x]*DIM), axis=-1).reshape(DIM, -1)
   epred = gpeh1.predict(xxx.T)
   epred2 = gpeh2.predict(xxx.T)
   newx2 = np.round(xxx[:, np.argmax(epred)], 5)
   newx1 = np.round(xxx[:, np.argmax(epred2)], 5)
   
   # Check convergence
   #  (assumes LF costs 100x HF)
   print("MAX IS (", np.max(ehid), np.max(epred), ') (', np.max(ehi1), np.max(epred2), ')')
   print(x2)
   #print("MAX IS ", np.max([ehi1 / .01, ehid]))
   if np.max([ehi1, ehid]) < 1e-2: break # (low-fidelity EHI is not weighted for stopping condition)

   # add next point according to weighted EHI
   if np.max(ehi1) / 0.01 > np.max(ehid):
      #if s[np.argmax(ehi1)] ==0: hey
      x2 = np.append(x2, np.atleast_2d(newx2), 0)
      #x2 = np.append(x2, np.atleast_2d(xx[:, np.argmax(ehi1)]), 0)
   else:
      #x1 = np.append(np.atleast_2d(xx[:, np.argmax(ehid)]), x1, 0)
      #x2 = np.append(np.atleast_2d(xx[:, np.argmax(ehid)]), x2, 0)
      x1 = np.append(np.atleast_2d(newx1), 0)
      x2 = np.append(np.atleast_2d(newx1), 0)

l = np.linspace(XL, XU, 60)
xx = np.meshgrid(l, l, l, l)[0].reshape(DIM, l.size ** (DIM) // DIM)
p1, s1 = gp1.predict(np.atleast_2d(xx).T, return_std=True)
p2, s2 = gp2.predict(np.atleast_2d(xx).T, return_std=True)
pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
pd2, sd2 = gpdelta2.predict(np.atleast_2d(xx).T, return_std=True)

# record final solution
fl = open('costLogMD.log', 'w')
fl.write('model evals\n')
fl.write('high %i\n' % (x1.size // DIM))
fl.write('low %i\n' % (x2.size // DIM))
xsol = xx[:, np.argmin(pd + p1)]
fl.write('Best Power: %s (%s)\n' % (str(f([xsol], lf=False)), str(xsol)))
fl.close()


