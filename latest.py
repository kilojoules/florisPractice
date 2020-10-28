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
    

# global parameters
pointdic = {}
DIM = 1
XI = 0.1
NEVALS = 0
n_initial = 2
XL, XU = (-30, 30)
FULLBAYES = True
KNOWLEDGE = True
OPTIMIZER = None
#OPTIMIZER = 'fmin_l_bfgs_b'
bounds = (np.ones(DIM) * XL, np.ones(DIM) * XU)
np.random.seed(832381)
DOMIN = False

# initial samples
initial_samps = [np.random.uniform(XL, XU, size=DIM) for _ in range(n_initial)]

# start log file
outf = open('log.log', 'w')
outf.write('y1 y2 y3 y4 pow\n')

# add initial samples to observed points
for point in initial_samps:
    pointdic[' '.join((str(s) for s in point))] = delta(point)
    #pointdic[' '.join((str(s) for s in point))] = f(point)
    outf.write(' '.join(
              [str(s) for s in point] + 
              [str(pointdic[' '.join((str(s) for s in point))])] + 
              ['\n']
              ))
    NEVALS += 1
outf.close()

# iterate through BO
for __ in range(120):

   # open log file
   outf = open('log.log', 'a')

   # summon all already-evaluated points
   thesePoints, theseEvals = [], []
   for point in pointdic.keys():
      thesePoints.append(float(point))
      #thesePoints.append(np.array([float(s) for s in point.split(' ')]))
      theseEvals.append(pointdic[point])
   thesePoints = np.atleast_2d(thesePoints).T
   X_sample, Y_sample = (np.array(thesePoints), np.array(theseEvals))

   # set kernel
   kernel = RBF(15.7 , (.3 , 5e2 )) 

   # create Gaussian Processes for objectives 1 and 2
   gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=OPTIMIZER)
   gp1.fit((thesePoints), np.array(theseEvals)[:, 0])
   gp2.fit((thesePoints), np.array(theseEvals)[:, 1])

   # create helper functions to simplify future GP calls
   def gpf1(x, return_std=False):
      alph, astd = gp1.predict(np.atleast_2d(x).T, return_std=True)
      if return_std:
         return (alph, astd)
      else:
         return alph 
   def gpf2(x, return_std=False):
      alph, astd = gp2.predict(np.atleast_2d(x).T, return_std=True)
      if return_std:
         return (alph, astd)
      else:
         return alph 

   # compute EI pareto front
   gridvals, EIvals, ParLoc = parEI(gpf1, gpf2, X_sample, Y_sample)
   if DOMIN:
      parX = np.array([gridvals[ParLoc][np.argmax(np.sqrt(np.sum(EIvals[:, ParLoc] ** 2, 0)))]])
   else:
      parX = np.array([gridvals[ParLoc][np.argmin(np.sqrt(np.sum(EIvals[:, ParLoc] ** 2, 0)))]])
   maxval = np.max((np.sqrt(np.sum(EIvals[:, ParLoc] ** 2, 0))))
   EIFront = EIvals[:, ParLoc]


   # plot 1D results
   if True:

      # set up figure
      fig, ax = plt.subplots(2, 2, figsize=(8, 8))
      plt.subplots_adjust(wspace=.3)

      # initialize plotting range
      x = np.linspace(XL, XU, 302)[1:-1]

      # summon evaluated points
      keys = pointdic.keys()
      keys = [str(key) for key in keys]

      #########
      #  top left
      ##########

      # make predictions of objectives
      gs1 = np.array([f(np.array([xc]), lf=True)[0] + gpf1(np.ones(DIM) * xc) for xc in x])[:, 0]
      gs2 = np.array([f(np.array([xc]), lf=True)[1] + gpf2(np.ones(DIM) * xc) for xc in x])[:, 0]
      gstd1 = np.array([gpf1(np.ones(DIM) * xc, return_std=True)[1] for xc in x])[:, 0]
      gstd2 = np.array([gpf2(np.ones(DIM) * xc, return_std=True)[1] for xc in x])[:, 0]

      # plot predictions
      ax[0][0].fill_between(x, gs1 - 2 * gstd1, gs1 + 2 * gstd1, facecolor='gray', alpha=0.3)
      axx = ax[0][0].twinx()
      axx.fill_between(x, gs2 - 2 * gstd2, gs2 + 2 * gstd2, facecolor='lightgray', alpha=0.3)
      ax[0][0].plot(x, gs1, label='Prediction', c='red')
      axx.plot(x, gs2, label='Prediction', c='red', ls='--')

      # plot truth
      ax[0][0].plot(x, [turbF(xc) for xc in x], c='yellow', lw=1, label='High Fidelity')
      axx.plot(x, [g(xc) for xc in x], c='yellow', lw=1, label='High Fidelity', ls='--')
      ax[0][0].set_xlim(XL, XU)
      keypoints = np.array([float(k.split(' ')[0]) for k in keys])

      # scatter observed points
      ax[0][0].scatter(keypoints, np.array([f(np.array([point]), lf=True)[0] for point in keypoints]) + np.array([pointdic[key][0] for key in keys]), marker='*', s=15, c='green', lw=3)

      #########
      #  top right
      ##########

      # compute EI for objectives 1 and 2, as well as a compromise
      s = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf1)[0] for xc in x]
      s2 = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf2)[0] for xc in x]
      ax[0][1].plot(x, s, label=r'$EI(f_1)$', c='red')
      ax[0][1].plot(x, s2, label=r'$EI(f_2)$', ls='--', c='blue')
      ax[0][1].plot(x, np.max([np.sqrt(np.array(s) ** 2 + np.array(s2) ** 2), np.zeros(len(s))], 0), label=r'$\sqrt{\sum_i EI(f_i)^2}$', ls='-.', c='purple')
      ax[0][1].axvline(parX, ls='--', c='w')
      ax[0][1].legend(loc='upper right')

      #########
      #  bottom left
      ##########
   
      # scatter EI pareto front
      cc = ax[1][0].scatter(EIFront.T[:, 0], EIFront.T[:, 1], c=np.sqrt((EIFront.T[:, 0] ** 2 +  EIFront.T[:, 1] ** 2)), s=55)
      cbar = fig.colorbar(cc, ax=ax[1][0], format=ticker.FuncFormatter(fmt), orientation='horizontal')
      cbar.set_label(r'$\sqrt{\sum_i EI(f_i)^2}$')
      cbar.ax.set_yticklabels(cbar.ax.get_xticklabels(), rotation='vertical')
      ax[1][0].legend(loc='upper left')

      # scatter all observed EI values
      ax[1][0].scatter(EIvals.T[:, 0], EIvals.T[:, 1], c='blue', marker='s')

      #########
      #  bottom right
      ##########

      # scatter true pareto front
      truthGrid, truthFs, truthIndices = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False, truth=True)
      truthFront = truthFs[:, truthIndices]
      ax[1][1].scatter(truthFront.T[:, 0], truthFront.T[:, 1], c='red', marker='s')

      # scatter approximated Pareto front
      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False)
      d = b[:, c]
      ax[1][1].scatter(d.T[:, 0], d.T[:, 1], c='yellow')

      # make titles
      ax[0][1].set_title("Expected Improvement")
      ax[1][0].set_xlabel(r'$EI(f_1)$')
      ax[1][0].set_ylabel(r'$EI(f_2)$')
      ax[1][1].set_xlabel(r'$f_1$')
      ax[1][1].set_ylabel(r'$f_2$')
      if DOMIN:
         plt.suptitle(r"Minimum, %i High Fidelity Evaluations" % (NEVALS))
         plt.savefig('gpminMO%05d' % __)
      else: 
         plt.suptitle(r"Maximum, %i High Fidelity Evaluations" % (NEVALS))
         plt.savefig('gpmaxMO%05d' % __)
      plt.clf()
      plt.close('all')

   # evaluate and save new point
   pointdic[' '.join((str(s) for s in parX))] = delta(parX)
   thisX = parX
   NEVALS += 1
   outf.write(' '.join(
              [str(s) for s in thisX] + 
              [str(pointdic[' '.join((str(s) for s in thisX))])] + 
              ['\n']
              ))
   outf.close()

   # stopping condition
   if __ > 1 and maxval < 1e-3: break
   if __ > 8: break

