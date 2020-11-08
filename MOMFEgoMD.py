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
OPTIMIZER = None
#OPTIMIZER = 'fmin_l_bfgs_b'
COST = 0.01

# discrepency function 
def delta(x):
   return (np.array([turbF(x, lf=False, MD=True), g(x, lf=False)])
          - np.array([turbF(x, lf=True, MD=True), g(x, lf=True)]))

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf, MD=True), g(x, lf=lf)]

DELTA_LENGTH_LOW_BOUNDS = 30
kernel = RBF(15 , (1 , 200)) # LF kernel
kernel2 = RBF(15 , (DELTA_LENGTH_LOW_BOUNDS, 200)) # discrepency kernel
DIM = 2

x1 = np.random.uniform(XL, XU, (2, DIM)) # HF samples
x2 = np.array(list(x1) + list(np.random.uniform(XL, XU, (50, DIM)))) # LF samples

logf = open('Ego_Call_Log.logged', 'w')
logf.write('HFCalls LFCalls\n')
# begin optimization
for __ in range(2000):

   x2 = np.round(x2, 5)
   x1 = np.round(x1, 5)

   # summon model evaluations
   fHs = np.array([delta(np.array([xc]))[0] for xc in x1])
   fLs = np.array([f(np.array([xc]), lf=True)[0] for xc in x2])
   fHs2 = np.array([delta(np.array([xc]))[1] for xc in x1])
   fLs2 = np.array([f(np.array([xc]), lf=True)[1] for xc in x2])
   
   
   # Fit GPs
   gpdelta = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta.fit(np.atleast_2d(x1), fHs)
   gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp1.fit(np.atleast_2d(x2), fLs)
   gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gpdelta2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp2.fit(np.atleast_2d(x2), fLs2)
   gpdelta2.fit(np.atleast_2d(x1), fHs2)
   
   # create mesh
   x = np.linspace(XL, XU, 20)
   #xx = np.array([xc for xc in itertools.permutations(l, 4)]).T
   #xx = np.meshgrid(l, l, l, l)[0].reshape(DIM, l.size ** (DIM) // DIM)
   #np.random.seed(12)
   xx = np.random.uniform(XL, XU, (DIM, 500))
   #xx = np.stack(np.meshgrid(*[x]*DIM), axis=-1).reshape(-1, DIM).T
   
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
         #return np.array([mu1, s1])
         return np.array([mu1 + mud, s1])
      else: 
         mu1 = gp1.predict(np.atleast_2d(x), return_std=False)
         mud = gpdelta.predict(np.atleast_2d(x), return_std=False)
         #return mu1 
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
         #return np.array([mu1, s1])
         return np.array([mu1 + mud, s1])
      else: 
         mu1 = gp2.predict(np.atleast_2d(x), return_std=False)
         mud = gpdelta2.predict(np.atleast_2d(x), return_std=False)
         #return mu1 
         return mu1 + mud

   # compute EHVI for each point in grid
   ehi1 = np.array([EHI(xc, gpr1, gpr2, x2=x2.T, MD=DIM, NSAMPS=500) for xc in xx.T])
   ehid = np.array([EHI(xc, gpr, gpr2d, x2=x1.T, MD=DIM, NSAMPS=500) for xc in xx.T])

   if True:
      print("PROBE")
      fig, ax = plt.subplots(3, 4, figsize=(10, 10))
      plt.subplots_adjust(wspace=.3, hspace=0.5)
      x = np.linspace(XL, XU, 15)#[1:-1]
      y = np.array([x, x]).T
      X, Y = np.meshgrid(x, x)

      fs1 = np.array([np.array([f([[xc, yc]])[0] for xc in x]) for yc in x])
      gs11 = np.array([np.array([gp1.predict(np.atleast_2d([xc, yc]))[0] for xc in x]) for yc in x])
      gstd11 = np.array([np.array([gp1.predict(np.atleast_2d([xc, yc]), return_std=True)[1][0] for xc in x]) for yc in x])
      gs1 = np.array([np.array([gpr(np.atleast_2d([xc, yc]))[0] for xc in x]) for yc in x])
      gstd1 = np.array([np.array([gpr(np.atleast_2d([xc, yc]), return_std=True)[1][0] for xc in x]) for yc in x])
      ei1 = np.array([np.array([EHI([xc, yc], gpr, gpr2d, x2=x1.T, MD=DIM, NSAMPS=500) for xc in x]) for yc in x])

      fs2 = np.array([np.array([f([[xc, yc]])[1] for xc in x]) for yc in x])
      gs22 = np.array([np.array([gp2.predict(np.atleast_2d([xc, yc]))[0] for xc in x]) for yc in x])
      gstd22 = np.array([np.array([gp2.predict(np.atleast_2d([xc, yc]), return_std=True)[1][0] for xc in x]) for yc in x])
      gs2 = np.array([np.array([gpr2d(np.atleast_2d([xc, yc]))[0] for xc in x]) for yc in x])
      gstd2 = np.array([np.array([gpr2d(np.atleast_2d([xc, yc]), return_std=True)[1][0] for xc in x]) for yc in x])
      ei2 = np.array([np.array([EHI([xc, yc], gpr1, gpr2, MD=DIM, x2=x2.T, NSAMPS=500) for xc in x]) for yc in x])
      #ei2 = np.array([np.array([expected_improvement(np.atleast_2d([xc, yc]).T,  X_sample, Y_sample, gpf2)[0] for xc in x]) for yc in x])


      #gs1 = np.array([f((xc), lf=True)[0] + gpf1(np.atleast_2d(xc).T) for xc in x])[:, 0]
      #gs2 = np.array([f((xc), lf=True)[1] + gpf2(np.atleast_2d(xc).T) for xc in x])[:, 0]
      #gs2 = np.array([gpf_next(np.ones(DIM) * xc) for xc in x])
      #gstd1 = np.array([gpf1(np.atleast_2d(xc).T, return_std=True)[1] for xc in x])[:, 0]
      #gstd2 = np.array([gpf2(np.atleast_2d(xc).T, return_std=True)[1] for xc in x])[:, 0]
      #gstd2 = np.array([gpf_next(np.ones(DIM) * xc, return_std=True)[1] for xc in x])



      c = ax[0][0].contourf(X, Y, fs1, 13)
      fig.colorbar(c, ax=ax[0][0])
      c = ax[1][0].contourf(X, Y, fs2, 13)
      fig.colorbar(c, ax=ax[1][0])
      ax[0][0].set_title('True HF Power')
      ax[1][0].set_title('True HF Loading')
      

      c = ax[0][1].contourf(X, Y, gs1, 13)
      fig.colorbar(c, ax=ax[0][1])
      c = ax[1][1].contourf(X, Y, gs2, 13)
      fig.colorbar(c, ax=ax[1][1])
      ax[0][1].set_title('Approx $\mu(Power)$')
      ax[1][1].set_title('Approx $\mu(Loading)$')

      c = ax[0][2].contourf(X, Y, gstd1, 13, cmap=plt.cm.coolwarm)
      fig.colorbar(c, ax=ax[0][2])
      c = ax[1][2].contourf(X, Y, gstd2, 13, cmap=plt.cm.coolwarm)
      fig.colorbar(c, ax=ax[1][2])
      ax[0][2].set_title('Approx $\sigma(Power)$')
      ax[1][2].set_title('Approx $\sigma(Loading)$')
      c = ax[0][3].contourf(X, Y, ei1, 13)
      fig.colorbar(c, ax=ax[0][3])
      ax[0][3].set_title('$EHVI$')
      c = ax[1][3].contourf(X, Y, ei2 / COST, 13)
      fig.colorbar(c, ax=ax[1][3])
      ax[1][3].set_title('$EHVI_1/%f$' % COST)
      #ax[1][3].set_title('$EI(f_2)$')

      c = ax[2][0].contourf(X, Y, gs11, 13, cmap=plt.cm.viridis)
      fig.colorbar(c, ax=ax[2][0])
      ax[2][0].set_title('Approx $\mu(Power^{LF})$')

      c = ax[2][1].contourf(X, Y, gstd11, 13, cmap=plt.cm.coolwarm)
      fig.colorbar(c, ax=ax[2][1])
      ax[2][1].set_title('Approx $\sigma(Power^{LF})$')

      c = ax[2][2].contourf(X, Y, gs22, 13, cmap=plt.cm.viridis)
      fig.colorbar(c, ax=ax[2][2])
      ax[2][2].set_title('Approx $\mu(Loading^{LF})$')

      c = ax[2][3].contourf(X, Y, gstd22, 13, cmap=plt.cm.coolwarm)
      fig.colorbar(c, ax=ax[2][3])
      ax[2][3].set_title('Approx $\sigma(Loading^{LF})$')

      for oo in range(4):
         for qq in range(2):
            if oo == 3 and qq == 1: 
               ax[qq][oo].scatter(x2[:, 0], x2[:, 1], marker='^', s=15, lw=3, c='pink')
               continue
            ax[qq][oo].scatter(x1[:, 0], x1[:, 1], marker='*', s=15, lw=3, c='lightblue')
      for oo in range(4):
         for qq in [2]:
            ax[qq][oo].scatter(x2[:, 0], x2[:, 1], marker='^', s=15, lw=3, c='pink')



      #a, b, c = parEI(gpr, gpr2d, '', '', EI=False, truth=True, MD=DIM)
      #d = b[:, c]
      #c = ax[1][2].scatter(d.T[:, 0], d.T[:, 1], c='red', marker='s')

      #a, b, c = parEI(gpr, gpr2d, '', '', EI=False, MD=DIM)
      #d = b[:, c]
      #c = ax[1][2].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))
      #cb = fig.colorbar(c, ax=ax[2][0])
      #cb.set_label(r'$\sum_i f_i^j$')

      #a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=True)
      #d = b[:, c]
      #c = ax[2][3].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))
      #cb = fig.colorbar(c, ax=ax[2][3])
      #cb.set_label(r'$\sum_i EI(f_i)^j$')
      #ax[2][0].set_xlabel(r'$f_1$')
      #ax[2][0].set_ylabel(r'$f_2$')
      #ax[2][3].set_xlabel(r'$-EI(f_1)$')
      #ax[2][3].set_ylabel(r'$-EI(f_2)$')

      #plt.suptitle(r"Minimize of $\sum_i EI_i^j$ %i High Fidelity Evaluations" % (NEVALS))
  #    ax[2][0].remove()
  #    ax[2][2].remove()
  #    ax[2][3].remove()
      plt.suptitle('%i HF Samples, %i LF Samples' % (x1.shape[0], x2.shape[0]))
      plt.savefig('gpEgo_again_100cost_1min_%05d' % __)
      plt.clf()
      plt.close('all')


   
   # Check convergence
   #  (assumes LF costs 100x HF)
   #print("MAX IS ", np.max(ei1), np.max(ei2) / .01)
   print("MAX IS ", np.max(ehid), np.max(ehi1 / COST))
   print("Plotted MAX IS ", np.max(ei1), np.max(ei2 / COST))
   logf.write('%i %i\n' % (x1.shape[0], x2.shape[0]))
   #print("MAX IS ", np.max([ehi1 / .01, ehid]))
   if np.max([ehid]) < 1e-2: break # (low-fidelity EHI is not weighted for stopping condition)
   #if np.max([ehid]) < 1e-1: break # (low-fidelity EHI is not weighted for stopping condition)

   # add next point according to weighted EHI
   #if np.max(ei2) / 0.01 > np.max(ei1):
   if np.max(ehi1) / COST > np.max(ehid):
      #if s[np.argmax(ehi1)] ==0: hey
      #x2 = np.append(x2, np.atleast_2d(xx[:, np.argmax(ei2)]), 0)
      x2 = np.append(x2, np.atleast_2d(xx[:, np.argmax(ehi1)]), 0)
      #x1 = np.append(np.atleast_2d(xx[:, np.argmax(ehid)]), x1, 0)
      #x2 = np.append(np.atleast_2d(xx[:, np.argmax(ehi1)]), x2, 0)

   else:
      x1 = np.append(np.atleast_2d(xx[:, np.argmax(ehid)]), x1, 0)
      x2 = np.append(np.atleast_2d(xx[:, np.argmax(ehid)]), x2, 0)
      #x1 = np.append(np.atleast_2d(xx[:, np.argmax(ei1)]), x1, 0)
      #x2 = np.append(np.atleast_2d(xx[:, np.argmax(ei1)]), x2, 0)

logf.close()
l = np.linspace(XL, XU, 62)
xx = np.meshgrid(l, l)[0].reshape(DIM, l.size ** (DIM) // DIM)
p1, s1 = gp1.predict(np.atleast_2d(xx).T, return_std=True)
p2, s2 = gp2.predict(np.atleast_2d(xx).T, return_std=True)
pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
pd2, sd2 = gpdelta2.predict(np.atleast_2d(xx).T, return_std=True)
# record final solution
fl = open('costLogMD22213.log', 'w')
fl.write('model evals\n')
fl.write('high %i\n' % (x1.size // DIM))
fl.write('low %i\n' % (x2.size // DIM))
xsol = xx[:, np.argmin(pd + p1)]
fl.write('Best Power: %s (%s)\n' % (str(f([xsol], lf=False)), str(xsol)))
fl.close()
