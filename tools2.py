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
XL, XU = (0, 30)
DIM=1

# formatting funciton
def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

# discrepency function (assumes no cost to LF model?)
def delta(x):
   return (np.array([turbF(x, lf=False)[0], g(x, lf=False)]) 
          - np.array([turbF(x, lf=True)[0], g(x, lf=True)]))

# objective functions
def f(x, lf=False):
   return [turbF(x, lf=lf)[0], g(x, lf=lf)]

# callback function
def callb(x): print('--- >', x)

# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient




def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr(X, return_std=True)
    mu_sample = np.array([gpr(xx)[0] for xx in X_sample])
    sigma = sigma.reshape(-1, 1)[:, 0]

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.min(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei =  imp * norm.cdf(Z) - sigma * norm.pdf(Z)
        ei[sigma <= 1e-8] = 0.0

    return np.min([ei, np.zeros(ei.shape)], 0)

'''
compute expected improvement pareto front
  EI - compute EI if True, f if false
  Truth - if not EI and true, querry truth model
          if not EI and False, querry surrogate model
  
  returns grid of potential inputs, the associated outputs, and indices associated with pareto front
'''
def parEI(gp1, gp2, X_sample, Y_sample, EI=True, truth=False):
    x = np.linspace(XL, XU, 100)
    ins = (x)
    #ins = np.atleast_2d(x).T
    #evs = gp.predict(ins)
    if EI:
       eis = expected_improvement(ins, X_sample, Y_sample[:, 0], gpf1)
       eis2 = expected_improvement(ins, X_sample, Y_sample[:, 1], gpf2)
       pars = is_pareto_efficient_simple(np.array([eis, eis2]).T)
       return (ins, np.array([eis, eis2]), pars)
    else:
       if not truth:
          a = [f(np.array([xc]), lf=True)[0] + gpf1(np.atleast_2d(np.array([xc])))[0] for xc in ins.T]
              # gpf1(np.atleast_2d(ins.T))
          b = [f(np.array([xc]), lf=True)[1] + gpf2(np.atleast_2d(np.array([xc])))[0] for xc in ins.T]
          #b = gpf2(np.atleast_2d(ins.T))
       else: 
          a = [turbF(i) for i in x]
          b = [g(i) for i in x]
       print(a) 
       pars = is_pareto_efficient_simple(np.array([a, b]).T)
       return(ins, np.array([a, b]), pars)
    


def MFKG(z, lfpoints, lfevals, hfpoints, hfevals, kernel, OPTIMIZE=True):

      # construct initial GP
      gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
      gpd = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
      gp1.fit(np.atleast_2d(lfpoints).T, lfevals)
      gpd.fit(np.atleast_2d(hfpoints).T, hfevals)
      def gp(x, return_std=False):
         return(np.array(gp1.predict(x, return_std=return_std)) + np.array(gpd.predict(x, return_std=return_std)))

      # Find initial minimum value from GP model
      min_val = 1e100
      X_sample = pnts
      Y_sample = evls
      #for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(20)]:
      res = mini(gp, x0=x0, bounds=[(0, 3) for ss in range(DIM)], method='Nelder-Mead')
         #res = mini(expected_improvement, x0=x0[0], bounds=[(XL, XU) for ss in range(DIM)], args=(X_sample, Y_sample, gp))#, callback=callb) 
      #   if res.fun < min_val:
      min_val = res.fun
      min_x = res.x

      # estimate min(f^{n+1}) with MC simulation
      MEAN = 0
      NSAMPS = 20
      for pp in range(NSAMPS):

         # construct future GP
         points = np.atleast_2d(np.append(X_sample, z)).T 
         m, s = gp(z, return_std=True)
         evals = np.append(evls, m + np.random.normal(0, s))
         gpnxt = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
         gpnxt.fit(points, evals)

         # convinience function
         def gpf_next(x, return_std=False):
            alph, astd = gpnxt.predict(np.atleast_2d(x), return_std=True)
            alph = alph[0]
            if return_std:
               return (alph, astd)
            else:
               return alph


         # search for minimum in future GP
         min_next_val = 99999
         #for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(10)]:
         res = mini(gpf_next, x0=x0, bounds=[(XL, XU) for ss in range(DIM)])
            #res = mini(expected_improvement, x0=x0, bounds=[(XL, XU) for ss in range(DIM)], args=(np.array(points), np.array(evals), gpf_next)) 
            #res = mini(gpf_next, x0=x0, bounds=[(0, 3) for ss in range(DIM)], args=(X_sample, Y_sample, gpf_next))
            #print('--> ', res.fun, res.fun[0] < min_next_val)
         #   if res.fun < min_next_val:
         min_next_val = res.fun
         min_next_x = res.x

         MEAN += min_next_val
      MEAN /= NSAMPS
      return min_val - MEAN



def KG(z, evls, pnts, gp, kernel):

      # Find initial minimum value from GP model
      min_val = 1e100
      X_sample = pnts
      Y_sample = evls
      #for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(20)]:
      x0 = np.random.uniform(XL, XU, size=DIM)
      res = mini(gp, x0=x0, bounds=[(XL, XU) for ss in range(DIM)]) #, method='Nelder-Mead')
         #res = mini(expected_improvement, x0=x0[0], bounds=[(XL, XU) for ss in range(DIM)], args=(X_sample, Y_sample, gp))#, callback=callb) 
      #   if res.fun < min_val:
      min_val = res.fun
      min_x = res.x
 

      # estimate min(f^{n+1}) with MC simulation
      MEAN = 0
      NSAMPS = 5000
      for pp in range(NSAMPS):

         # construct future GP, using z as the next point
         points = np.atleast_2d(np.append(X_sample, z)).T  
         m, s = gp(z, return_std=True)
         evals = np.append(evls, m + np.random.normal(0, s))
         gpnxt = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
         gpnxt.fit(points, evals)

         # convinience function
         def gpf_next(x, return_std=False):
            alph, astd = gpnxt.predict(np.atleast_2d(x), return_std=True)
            alph = alph[0]
            if return_std:
               return (alph, astd)
            else:
               return alph
 
         # search for minimum in future GP
         min_next_val = 99999
         for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(10)]:
            res = mini(gpf_next, x0=x0, bounds=[(XL, XU) for ss in range(DIM)])
            #res = mini(expected_improvement, x0=x0, bounds=[(XL, XU) for ss in range(DIM)], args=(np.array(points), np.array(evals), gpf_next)) 
            #res = mini(gpf_next, x0=x0, bounds=[(0, 3) for ss in range(DIM)], args=(X_sample, Y_sample, gpf_next))
            #print('--> ', res.fun, res.fun[0] < min_next_val)
         #   if res.fun < min_next_val:
            print('+++++++++ ', res.fun)
         hey
         min_next_val = res.fun
         min_next_x = res.x

         MEAN += min_next_val
      MEAN /= NSAMPS
      return min_val - MEAN



def parEI(gp1, gp2, X_sample, Y_sample, EI=True, truth=False):
    x = np.linspace(XL, XU, 100)
    ins = (x)
    #ins = np.atleast_2d(x).T
    #evs = gp.predict(ins)
    if EI:
       #eis = expected_improvement(ins, X_sample, Y_sample[:, 0], gpf1)
       #eis2 = expected_improvement(ins, X_sample, Y_sample[:, 1], gpf2)
       eis = np.array([KG(ns, X_sample, Y_sample[:, 0], gpf1) for ns in ins])
       eis2 = np.array([KG(ns, X_sample, Y_sample[:, 1], gpf2) for ns in ins])
       pars = is_pareto_efficient_simple(np.array([eis, eis2]).T)
       return (ins, np.array([eis, eis2]), pars)
    else:
       if not truth:
          a = [f(np.array([xc]), lf=True)[0] + gpf1(np.atleast_2d(np.array([xc])))[0] for xc in ins.T]
              # gpf1(np.atleast_2d(ins.T))
          b = [f(np.array([xc]), lf=True)[1] + gpf2(np.atleast_2d(np.array([xc])))[0] for xc in ins.T]
          #b = gpf2(np.atleast_2d(ins.T))
       else: 
          a = [turbF(i) for i in x]
          b = [g(i) for i in x]
       #print(a) 
       pars = is_pareto_efficient_simple(np.array([a, b]).T)
       return(ins, np.array([a, b]), pars)
    

