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

kernel = RBF(15 , (9 , 5e2))
DIM = 2


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
   xx = np.random.uniform(XL, XU, (2, 400))
   
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
   
   print('probe', __)
   ehid = np.array([EHI(xc, gpr, gpr2d, MD=DIM, NSAMPS=500) for xc in xx.T])
   #eid = EHI(15, gpr, gpr1)
   #ei1 = -1 * expected_improvement(xx, x2, fHs + fLs[:fHs.size], gpr1, xi=XI)
   #eid = -1 * expected_improvement(xx, x1, fHs + fLs[:fHs.size], gpr, xi=0.01)
   #ax[2, 0].plot(xx, eid, label=r'$EI(\mu_\delta)$', c='r')

   print(np.max([ehid]))


   if True:
      print("PROBE")
      fig, ax = plt.subplots(2, 4, figsize=(10, 10))
      plt.subplots_adjust(wspace=.5, hspace=0.5)
      x = np.linspace(XL, XU, 15)#[1:-1]
      y = np.array([x, x]).T
      X, Y = np.meshgrid(x, x)

      fs1 = np.array([np.array([f([[xc, yc]])[0] for xc in x]) for yc in x])
      gs1 = np.array([np.array([gpr(np.atleast_2d([xc, yc]))[0] for xc in x]) for yc in x])
      gstd1 = np.array([np.array([gpr(np.atleast_2d([xc, yc]), return_std=True)[1][0] for xc in x]) for yc in x])
      ei1 = np.array([np.array([EHI([xc, yc], gpr, gpr2d, MD=DIM, NSAMPS=500) for xc in x]) for yc in x])

      fs2 = np.array([np.array([f([[xc, yc]])[1] for xc in x]) for yc in x])
      gs2 = np.array([np.array([gpr2d(np.atleast_2d([xc, yc]))[0] for xc in x]) for yc in x])
      gstd2 = np.array([np.array([gpr2d(np.atleast_2d([xc, yc]), return_std=True)[1][0] for xc in x]) for yc in x])
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
      #c = ax[1][3].contourf(X, Y, -1 * ei2, 13)
      #fig.colorbar(c, ax=ax[1][3])
      ax[0][3].set_title('$EHVI$')
      #ax[1][3].set_title('$EI(f_2)$')

      for qq in range(2):
         for oo in range(4):
            if qq == 1 and oo == 3: continue
            ax[qq][oo].scatter(x1[:, 0], x1[:, 1], marker='x', s=15, lw=1, c='k')



      a, b, c = parEI(gpr, gpr2d, x1.T, '', EI=False, truth=True, MD=DIM)
      d = b[:, c]
      c = ax[1][3].scatter(d.T[:, 0], d.T[:, 1], c='red', marker='s', label='Truth')

      a, b, c = parEI(gpr, gpr2d, x1.T, '', EI=False, MD=DIM)
      d = b[:, c]
      c = ax[1][3].scatter(d.T[:, 0], d.T[:, 1], c='yellow', label='Approximation') # c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))
      ax[1][3].legend(prop={'size':5})
      ax[1][3].set_xlabel("Power")
      ax[1][3].set_ylabel("Loading")
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
      plt.suptitle('%i HF Samples' % (__ + 2))
      plt.savefig('gpMO%05d' % __)
      plt.clf()
      plt.close('all')

   if np.max([ehid]) < 1e-2: break
   x1 = np.append(np.atleast_2d(xx[:, np.argmax(ehid)]), x1, 0)

l = np.linspace(XL, XU, 60)
xx = np.meshgrid(l, l)[0].reshape(DIM, l.size ** (DIM) // DIM)
pd, sd = gpdelta.predict(np.atleast_2d(xx).T, return_std=True)
pd2, sd2 = gpdelta2.predict(np.atleast_2d(xx).T, return_std=True)

fl = open('BOcostLogMD.log', 'w')

fl.write('model evals\n')
fl.write('high %i\n' % (x1.size // DIM))
xsol = xx[:, np.argmin(pd)]
fl.write('Best Power: %s (%s)\n' % (str(f([xsol], lf=False)), str(xsol)))
fl.close()

