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
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

plt.style.use('dark_background')
np.random.seed(17)
OPTIMIZER = 'fmin_l_bfgs_b'
COST = 0.01 # LF/HF cost

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf), g(x, lf=lf)]

# Initial samples
x1 = np.random.uniform(XL, XU, 2)
x2 = np.array(list(x1) + list(np.random.uniform(XL, XU, 1)))

# begin optimization
for __ in range(2000):

   # Summon funciton evaluations
   fHs = np.array([f(xc, lf=False)[0] for xc in x1])
   fLs = np.array([f(xc, lf=True)[0] for xc in x2])
   fHs2 = np.array([f(xc, lf=False)[1] for xc in x1])
   fLs2 = np.array([f(xc, lf=True)[1] for xc in x2])
   
   # convert evaluations to Emukit format   
   X_train, Y_train = convert_xy_lists_to_arrays([np.atleast_2d(x2).T, np.atleast_2d(x1).T], [np.atleast_2d(fLs).T, np.atleast_2d(fHs).T])
   X_train2, Y_train2 = convert_xy_lists_to_arrays([np.atleast_2d(x2).T, np.atleast_2d(x1).T], [np.atleast_2d(fLs2).T, np.atleast_2d(fHs2).T])

   # create MF kernels for objectives 1 and 2
   kernels = [GPy.kern.RBF(1, lengthscale=10), GPy.kern.RBF(1, lengthscale=10)]
   kernels2 = [GPy.kern.RBF(1, lengthscale=10), GPy.kern.RBF(1, lengthscale=10)]
   lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
   lin_mf_kernel2 = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels2)

   # Fit Emukit models

   # objective 1
   gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
   gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
   gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
   lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)
   lin_mf_model.optimize()

   # objective 2
   gpy_lin_mf_model2 = GPyLinearMultiFidelityModel(X_train2, Y_train2, lin_mf_kernel2, n_fidelities=2)
   gpy_lin_mf_model2.mixed_noise.Gaussian_noise.fix(0)
   gpy_lin_mf_model2.mixed_noise.Gaussian_noise_1.fix(0)
   lin_mf_model2 = model2 = GPyMultiOutputWrapper(gpy_lin_mf_model2, 2, n_optimization_restarts=5)
   lin_mf_model2.optimize()

   # define helper functions 
   # (returns HF prediction mean, HF or LF variance)
   def gpr(x, return_std=False):
      xh = np.append(np.atleast_2d(x).T, np.atleast_2d(np.ones(x.size)).T, 1)
      mu, vd = lin_mf_model.predict(xh)
      std = np.sqrt(vd)
      if return_std:
         return np.array([mu[:, 0], std[:, 0]])
      else: 
         return mu[:, 0]
   
   def gpr1(x, return_std=False):
      xl = np.append(np.atleast_2d(x).T, np.atleast_2d(np.zeros(x.size)).T, 1)
      mu, v1 = lin_mf_model.predict(xl)
      std = np.sqrt(v1)
      xh = np.append(np.atleast_2d(x).T, np.atleast_2d(np.ones(x.size)).T, 1)
      mu, vd = lin_mf_model.predict(xh)
      if return_std:
         return np.array([mu[:, 0], std[:, 0]])
      else: 
         return mu[:, 0]

   def gpr2d(x, return_std=False):
      xh = np.append(np.atleast_2d(x).T, np.atleast_2d(np.ones(x.size)).T, 1)
      xl = np.append(np.atleast_2d(x).T, np.atleast_2d(np.zeros(x.size)).T, 1)
      mu, v = lin_mf_model2.predict(xh)
      mu, v2 = lin_mf_model2.predict(xl)
      std = np.sqrt(v2)
      if return_std:
         return np.array([mu[:, 0], std[:, 0]])
      else: 
         return mu[:, 0]
   
   def gpr2(x, return_std=False):
      xh = np.append(np.atleast_2d(x).T, np.atleast_2d(np.ones(x.size)).T, 1)
      xl = np.append(np.atleast_2d(x).T, np.atleast_2d(np.zeros(x.size)).T, 1)
      mu, v2 = lin_mf_model2.predict(xl)
      std = np.sqrt(v2)
      mu, v2 = lin_mf_model2.predict(xh)
      if return_std:
         return np.array([mu[:, 0], std[:, 0]])
      else: 
         return mu[:, 0]

   # estimate EHVI
   xx = np.linspace(XL, XU, 100)
   ehi1 = np.array([EHI(xc, gpr1, gpr2, PCE=False) for xc in xx])
   ehid = np.array([EHI(xc, gpr, gpr2d, PCE=False) for xc in xx])

   # Make predictions for plotting
   xl = np.append(np.atleast_2d(xx).T, np.atleast_2d(np.zeros(xx.size)).T, 1)
   xh = np.append(np.atleast_2d(xx).T, np.atleast_2d(np.ones(xx.size)).T, 1)
   p1, v1 = lin_mf_model.predict(xl)
   p1 = p1[:, 0]
   s1 = np.sqrt(v1)[:, 0]
   pd, vd = lin_mf_model.predict(xh)
   pd = pd[:, 0]
   sd = np.sqrt(vd)[:, 0]
   p2, v2 = lin_mf_model2.predict(xl)
   p2 = p2[:, 0]
   s2 = np.sqrt(v2)[:, 0]
   pd2, vd2 = lin_mf_model2.predict(xh)
   pd2 = pd2[:, 0]
   sd2 = np.sqrt(vd2)[:, 0]

   # create plots
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
   ax12.fill_between(xx, pd2 - s2 - sd2, pd2 + s2 + sd2, facecolor='purple', alpha=0.7)
   ax12.fill_between(xx, pd2 - s2, pd2 + s2, facecolor='lightblue', alpha=0.7)
   ax[1, 0].fill_between(xx, pd - s1 - sd, pd + s1 + sd, facecolor='red', alpha=0.7)
   ax[1, 0].fill_between(xx, pd - s1, pd + s1, facecolor='lightblue', alpha=0.7)
   ax[1, 0].plot(xx, pd, c='red')
   ax[1, 0].plot(xx, [f(np.array([xc]), lf=False)[0] for xc in xx], c='yellow')
   ax[1, 0].scatter(x1, fHs, c='w', marker='x')
   ax12.plot(xx, pd2, c='purple')
   ax12.plot(xx, [f(np.array([xc]), lf=False)[1] for xc in xx], c='yellow', ls='--')
   ax12.scatter(x1, fHs2, c='w', marker='x')
   ax[2, 0].set_xlabel('x')
   ax[0, 0].set_ylabel(r'Low-Fidelity')
   ax[1, 0].set_ylabel(r'High-Fidelity')
   ax[0, 1].set_visible(False)
   ax[1, 1].set_visible(False)
   a, b, c = parEI(gpr, gpr2d, x1, np.array([fHs, fHs2]), EI=False)
   ax[2, 1].scatter(b[:, c].T[:, 0], b[:, c].T[:, 1], c='red')
   a, b, c = parEI(lambda l: [f(l)[0]], lambda l: [f(l)[1]], x1, np.array([fHs, fHs2]), EI=False)
   ax[2, 1].scatter(b[:, c].T[:, 0], b[:, c].T[:, 1], c='yellow')
   ax[2, 0].plot(xx, ehi1 / COST, label='EHVI$(\mu_1) / %f$' % COST, c='lightblue')
   ax[2, 0].plot(xx, ehid, label='EHVI($\mu_\delta$)', c='red')
   ax[2, 0].legend()
   ax[2, 1].set_xlabel('Negative Power')
   ax[2, 1].set_ylabel('Load')
   plt.savefig('Emukit_EHVI_test_%03d' % __)
   plt.clf()

   # Determine if stopping condition is met
   if np.max(ehid) < 1e-4: break

   # Determine which model to sample next
   if np.max(ehi1) / COST > np.max(ehid):
      x2 = np.append(x2, xx[np.argmax(ehi1)])
   else:
      x1 = np.append(xx[np.argmax(ehid)], x1)
      x2 = np.append(xx[np.argmax(ehid)], x2)

# record results
fl = open('emukitcostLog.log', 'w')
fl.write('model evals\n')
fl.write('high %i\n' % x1.size)
fl.write('low %i\n' % x2.size)
fl.write('Best Power: %s (%s)\n' % (str(np.min(pd + p1)), str(xx[np.argmin(pd + p1)])))
fl.write('Best Load: %s (%s)\n' % (str(np.min(pd2 + p2)), str(xx[np.argmin(pd2 + p2)])))
fl.close()
