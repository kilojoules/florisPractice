import numpy as np
import chaospy as cp
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
import numba as nb
plt.style.use('dark_background')
#XL, XU = (0, 1)
XL, XU = (-30, 30)
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
#@nb.jit(nopython=False)
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

# http://krasserm.github.io/2018/03/21/bayesian-optimization/
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.0):
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
        #ei[sigma <= 1e-8] = 0.0

    #return ei
    return np.min([ei, np.zeros(ei.shape)], 0)

'''
compute expected improvement pareto front
  EI - compute EI if True, f if false
  Truth - if not EI and true, querry truth model
          if not EI and False, querry surrogate model
  
  returns grid of potential inputs, the associated outputs, and indices associated with pareto front
'''
#@nb.jit(nopython=False, forceobj=True)
def parEI(gp1, gp2, X_sample, Y_sample, EI=True, truth=False, MD=False, PAR_RES=100):


    if MD: 
       # create ND grid
       x = np.linspace(XL, XU, PAR_RES)
       ins = np.stack(np.meshgrid(*[x]*MD), axis=-1).reshape(MD, -1)
       if X_sample is not None: ins = np.append(ins, X_sample, 1)
    else:
       # create 1D grid
       ins = np.linspace(XL, XU, PAR_RES)

    if EI:
       # compute EI front
       eis = expected_improvement(ins, X_sample, Y_sample[:, 0], gpf1)
       eis2 = expected_improvement(ins, X_sample, Y_sample[:, 1], gpf2)
       pars = is_pareto_efficient_simple(np.array([eis, eis2]).T)
       return (ins, np.array([eis, eis2]), pars)
    else:
       if not truth:
          # compute mu_GP front
          if MD:
             a = gp1(ins.T)
             b = gp2(ins.T)
          else:
             a = [gp1(np.atleast_2d(np.array([xc])))[0] for xc in ins.T]
             b = [gp2(np.atleast_2d(np.array([xc])))[0] for xc in ins.T]
       else: 
          # compute truth front
          if MD:
             a = [turbF([i], MD=MD) for i in ins.T]
          else:
             a = [turbF(i, MD=MD) for i in ins.T]
          b = [g(i) for i in ins.T]
       pars = is_pareto_efficient_simple(np.array([a, b]).T)

       return(ins, np.array([a, b]), pars)
    


# Knowledge gradient computation (not used)
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




def KG(z, evls, pnts, gp, kernel, NSAMPS=30, DEG=3, sampling=False):

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
      points = np.atleast_2d(np.append(X_sample, z)).T  
      m, s = gp(z, return_std=True)
      distribution = cp.J(cp.Normal(0, s))
      samples = distribution.sample(NSAMPS, rule='Halton')
      PCEevals = []
      for pp in range(NSAMPS):

         # construct future GP, using z as the next point
         evals = np.append(evls, m + samples[pp])
         #evals = np.append(evls, m + np.random.normal(0, s))
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
 
         res = mini(gpf_next, x0=x0, bounds=[(XL, XU) for ss in range(DIM)])
         min_next_val = res.fun
         min_next_x = res.x

         
         #print('+++++++++ ', res.fun)
         #MEAN += min_next_val
         PCEevals.append(min_next_val)
      if not sampling:
         polynomial_expansion = cp.orth_ttr(DEG, distribution)
         foo_approx = cp.fit_regression(polynomial_expansion, samples, PCEevals)
         MEAN = cp.E(foo_approx, distribution)
      else: MEAN = np.mean(PCEevals)
      #print(PCEevals, '...', MEAN)
      #hey
      #MEAN /= NSAMPS
      return min_val - MEAN



# 2-objective Hypervolume (area) Computation
#   First order approximation of pareto front area,
#   given nondominated reference point r_0
def H(fs, r=(0, 0)):
    fs = sorted(fs.tolist())
    summ = 0
    n = len(fs) - 1
    for ii in range(n):
       summ += (fs[ii][0] - fs[ii+1][0]) * (fs[ii][1] - r[1])
    summ += (fs[n][0] - r[0]) * (fs[n][1] - r[1])
    return summ

# Expected hypervolume computation
'''
MD - number of input dimensions if not 1
xi - see expected_improvement function definition
NSAMPS - number of random samples employed
PCE - Flag to use Polynomial Chaos Expansion (PCE)
ORDER - PCE order
'''
#@nb.jit(nopython=False)
def EHI(x, gp1, gp2, xi=0., x2=None, MD=None, NSAMPS=200, PCE=False, ORDER=2, PAR_RES=100):

    mu1, std1 = gp1(x, return_std=True)
    mu2, std2 = gp2(x, return_std=True)

    a, b, c = parEI(gp1, gp2, x2, '', EI=False, MD=MD, PAR_RES=PAR_RES)
    par = b.T[c, :]
    par += xi
    MEAN = 0 # running sum for observed hypervolume improvement
    if not PCE: # Monte Carlo Sampling
       for ii in range(NSAMPS):

          # add new point to Pareto Front
          evl = [np.random.normal(mu1, std1), np.random.normal(mu2, std2)]
          pears = np.append(par.T, evl, 1).T
          idx = is_pareto_efficient_simple(pears)
          newPar = pears[idx, :]

          # check if Pareto front improvemed from this point
          if idx[-1]:
             MEAN += H(newPar) - H(par)
   
       return(MEAN / NSAMPS) 
    else: 
       # Polynomial Chaos
       # (assumes 2 objective functions)
       distribution = cp.J(cp.Normal(0, std1), cp.Normal(0, std2))

       # sparse grid samples
       samples = distribution.sample(NSAMPS, rule='Halton')
       PCEevals = []
       for pp in range(NSAMPS):

          # add new point to Pareto Front
          evl = [np.random.normal(mu1, std1), np.random.normal(mu2, std2)]
          pears = np.append(par.T, evl, 1).T
          idx = is_pareto_efficient_simple(pears)
          newPar = pears[idx, :]

          # check if Pareto front improvemes
          if idx[-1]:
             PCEevals.append(H(newPar) - H(par))
          else:
             PCEevals.append(0)
       polynomial_expansion = cp.orth_ttr(ORDER, distribution)
       foo_approx = cp.fit_regression(polynomial_expansion, samples, PCEevals)
       MEAN = cp.E(foo_approx, distribution)
       return(MEAN)
