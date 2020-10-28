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

# two objectives
def f(x):
   #return [turbF(x)[0], turbF(x)[0]]
   return [turbF(x)[0], g(x)]

# given NxM matrix of objectives, generate pareto front
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


plt.style.use('dark_background')

# Compute expected improvement
# (assumpes f is negative)
def expected_improvement(X, X_sample, Y_sample, gpr, xi=.05):
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
    #mu_sample = np.array([gpr(xx)[0] for xx in X_sample])

    sigma = sigma.reshape(-1, 1)[:, 0]

    mu_sample_opt = np.min(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei =  imp * norm.cdf(Z) - sigma * norm.pdf(Z)
        ei[sigma <= 1e-8] = 0.0

    return np.min([ei, np.zeros(ei.shape)], 0)



# intialize book-keeping parameters
pointdic = {}
DIM = 1
XI = 0.1
NEVALS = 0
n_initial = 2
XL, XU = (-30, 30)
FULLBAYES = True
KNOWLEDGE = True
bounds = (np.ones(DIM) * XL, np.ones(DIM) * XU)
np.random.seed(1112121184)
initial_samps = [np.random.uniform(XL, XU, size=DIM) for _ in range(n_initial)]
outf = open('log.log', 'w')
outf.write('y1 y2 y3 y4 pow\n')



# accepts GPs and observations, returns ranked Pareto front
def parEI(gp1, gp2, X_sample, Y_sample, EI=True, knowgrad=False, truth=False):
    x = np.linspace(XL, XU, 100)
    ins = (x)
    #ins = np.atleast_2d(x).T
    #evs = gp.predict(ins)
    if knowgrad:
       eis = [KG(xc, gpn=0) for xc in x]
       eis2 = [KG(xc, gpn=1) for xc in x]
       pars = is_pareto_efficient_simple(np.array([eis, eis2]).T)
       return (ins, np.array([eis, eis2]), pars)
    elif EI:
       eis = expected_improvement(ins, X_sample, Y_sample[:, 0], gpf1)
       eis2 = expected_improvement(ins, X_sample, Y_sample[:, 1], gpf2)
       pars = is_pareto_efficient_simple(np.array([eis, eis2]).T)
       return (ins, np.array([eis, eis2]), pars)
    else:
       if not truth:
          a = gpf1(np.atleast_2d(ins.T))
          b = gpf2(np.atleast_2d(ins.T))
       else: 
          a = [turbF(i) for i in x]
          b = [g(i) for i in x]
       pars = is_pareto_efficient_simple(np.array([a, b]).T)
       return(ins, np.array([a, b]), pars)
    

# sample initial points
for point in initial_samps:
    pointdic[' '.join((str(s) for s in point))] = f(point)
    outf.write(' '.join(
              [str(s) for s in point] + 
              [str(pointdic[' '.join((str(s) for s in point))])] + 
              ['\n']
              ))
    NEVALS += 1

outf.close()

# begin optimization
for __ in range(120):

   outf = open('log.log', 'a')

   # collect observed points
   thesePoints, theseEvals = [], []
   for point in pointdic.keys():
      thesePoints.append(float(point))
      #thesePoints.append(np.array([float(s) for s in point.split(' ')]))
      theseEvals.append(pointdic[point])
   theseEvals = np.array(theseEvals)

   # create GPs
   thesePoints = np.atleast_2d(thesePoints).T
   kernel = RBF(15.7 , (.3 , 5e2 )) #+ RBF(np.ones(DIM) * 1e-6, (1e-9, 1e-2))
   gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=None)
   gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=None)
   gp1.fit((thesePoints), np.array(theseEvals)[:, 0])
   gp2.fit((thesePoints), np.array(theseEvals)[:, 1])
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


   X_sample, Y_sample = (np.array(thesePoints), np.array(theseEvals))




   # knowledge gradient (for one process)
   # accpts x for constructing new GP
   def KG(x, expect=True, gpn=0):

      # select GP of interest
      if gpn == 0:
         gpf = gpf1
      elif gpn == 1:
         gpf = gpf2
      else: hey


      min_val = 1e100
      for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(20)]:
         #print(x0)
         #res = mini(gpf, x0=x0, bounds=[(0, 3) for ss in range(DIM)], method='Nelder-Mead')
         res = mini(expected_improvement, x0=x0[0], bounds=[(XL, XU) for ss in range(DIM)], args=(X_sample, Y_sample, gpf))
         if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
   #hey


      # construct next GP
      print('----> ', theseEvals)
      print('----> ', thesePoints)
      points = [s for s in list(thesePoints) + [np.array(x)]]
      evals = [s for s in list(theseEvals[:, gpn])] + [gpf(x)[0]]
      #evals = [s for s in list(theseEvals) + [gpf1(x)[0], gpf2(x)[0]]]
      #evals1 = [s[0] for s in list(theseEvals) + [gpf1(x)[0]]]
      #evals2 = [s[1] for s in list(theseEvals) + [gpf2(x)[0]]]
      gpnxt = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
      #gpnxt1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
      #gpnxt2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
      if DIM == 1:
         gpnxt.fit(np.atleast_2d(points).T, evals)
         #gpnxt1.fit(np.atleast_2d(points).T, evals[:, 0])
         #gpnxt2.fit(np.atleast_2d(points).T, evals[:, 1])
      else: 
         gpnxt.fit(points, evals)

      def gpf_next(x, return_std=False):
         alph, astd = gpnxt.predict(np.atleast_2d(x), return_std=True)
         if return_std:
            return (alph, astd)
         else:
            return alph

      # 
      #a, b, c = parEI(gpf_next1, gpf_next2, np.array(points), np.array(evals), knowgrad=True)
      min_next_val = 1
      for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(20)]:
         if not expect: 
            res = mini(gpf_next, x0=x0, bounds=[(0, 3) for ss in range(DIM)])
         else: 
            print(gpf_next(x0)) 
            res = mini(expected_improvement, x0=x0, bounds=[(XL, XU) for ss in range(DIM)], args=(np.array(points), np.array(evals), gpf_next)) 
         #res = mini(gpf_next, x0=x0, bounds=[(0, 3) for ss in range(DIM)], args=(X_sample, Y_sample, gpf_next))
         #print('--> ', res.fun, res.fun[0] < min_next_val)
         if res.fun < min_next_val:
            min_next_val = res.fun
            min_next_x = res.x

      print(gpf(min_x))
      print( gpf_next(min_next_x))
      print('++++++++++')
      print(expected_improvement(min_x, X_sample, Y_sample, gpf))
      print(expected_improvement(min_next_x, X_sample, Y_sample, gpf))
      print('-----')
      print(min_val, min_next_val)
      #if (expected_improvement(min_x, np.array(points), np.array(evals), gpf)[0] - expected_improvement(min_x, np.array(points), np.array(evals), gpf_next)[0]) > 0: hey
      if expect:
         return (expected_improvement(min_x, np.array(points), np.array(evals), gpf)[0] - expected_improvement(min_x, np.array(points), np.array(evals), gpf_next)[0])
      else:
         return (gpf(min_x)[0] - gpf_next(min_next_x))

   #if KNOWLEDGE:
   #   min_KG_val = 1
   #   for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(3)]:
   #      #print(x0)
   #      res = ps(KG, x0=x0, bounds=[(XL, XU) for ss in range(DIM)], deltaX=10)
   #      if res['f'] < min_KG_val:
   #         min_KG_val = res['f']
   #         min_KG_x = res['x']





   a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, knowgrad=True)
   d = b[:, c]
   parX = np.array([a[c][np.argmax(np.sqrt(np.sum(b[:, c] ** 2, 0)))]])


   if False: 
      plt.clf()
      plt.close('all')
      inx = np.linspace(XL, XU, 1000)
      m1 = np.array([gpf(xc)[0] for xc in inx])[:, 0]
      m2 = np.array([gpf_next(xc)[0] for xc in inx])[:, 0]
      s2 = np.array([gpf_next(xc, return_std=True)[1] for xc in inx])[:, 0]
      s1 = np.array([gpf(xc, return_std=True)[1] for xc in inx])[:, 0]
      print(s1.shape, m1.shape)
      plt.fill_between(inx, m1 - 2 * s1, m1 + 2 * s1, facecolor='red', alpha=.2)
      plt.fill_between(inx, m2 - 2 * s2, m2 + 2 * s2, facecolor='blue', alpha=.2)
      plt.scatter(min_x, gpf(min_x), c='red')
      plt.scatter(min_next_x, gpf_next(min_next_x)[0], c='blue', marker='x')
      plt.savefig('hey/%.3f___%.5f.png' % (x, gpf(min_x)[0] - gpf_next(min_next_x)))

   if True:
      print("PROBE")
      print(pointdic)
      fig, ax = plt.subplots(2, 2, figsize=(6, 6))
      x = np.linspace(XL, XU, 302)[1:-1]
      keys = pointdic.keys()
      keys = [str(key) for key in keys]
      gs1 = np.array([gpf1(np.ones(DIM) * xc) for xc in x])[:, 0]
      gs2 = np.array([gpf2(np.ones(DIM) * xc) for xc in x])[:, 0]
      #gs2 = np.array([gpf_next(np.ones(DIM) * xc) for xc in x])
      gstd1 = np.array([gpf1(np.ones(DIM) * xc, return_std=True)[1] for xc in x])[:, 0]
      gstd2 = np.array([gpf2(np.ones(DIM) * xc, return_std=True)[1] for xc in x])[:, 0]
      #gstd2 = np.array([gpf_next(np.ones(DIM) * xc, return_std=True)[1] for xc in x])
      ax[0][0].fill_between(x, gs1 - 2 * gstd1, gs1 + 2 * gstd1, facecolor='gray', alpha=0.3)
      axx = ax[0][0].twinx()
      axx.fill_between(x, gs2 - 2 * gstd2, gs2 + 2 * gstd2, facecolor='lightgray', alpha=0.3)
      #ax[0].fill_between(x, gs2 - 2 * gstd2, gs2 + 2 * gstd2, facecolor='purple', alpha=0.6)
      ax[0][0].plot(x, gs1, label='Prediction', c='red')
      axx.plot(x, gs2, label='Prediction', c='red', ls='--')
      #ax[0].plot(x, [gpf_next(np.ones(DIM) * xc)[0] for xc in x], label='Next Prediction', c='purple', ls='--')
      #plt.plot(x, g(x) + [gpf(np.ones(DIM) * xc)[0] for xc in x])
      ax[0][0].plot(x, [turbF(xc) for xc in x], c='b', lw=1, label='High Fidelity')
      #axx.plot(x, [turbF(xc) for xc in x], c='yellow', lw=1, label='High Fidelity')
      axx.plot(x, [g(xc) for xc in x], c='b', lw=1, label='High Fidelity', ls='--')
      ax[0][0].set_xlim(XL, XU)
      ax[0][0].scatter([float(k.split(' ')[0]) for k in keys], [pointdic[key][0] for key in keys], marker='*', s=15, c='green', lw=3)
      axx.scatter([float(k.split(' ')[0]) for k in keys], [pointdic[key][1] for key in keys], marker='*', s=15, c='lightgreen', lw=3, ls='--')

      s = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf1)[0] for xc in x]
      s2 = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf2)[0] for xc in x]
      #ax[1].plot(x, np.max([s, np.zeros(len(s))], 0) )
      spo = [float(k.split(' ')[0]) for k in keys]
      ax[0][1].plot(x, s, label='EI', c='b')
      ax[0][1].plot(x, s2, label='EI', ls='--', c='b')
      #ax[1].plot(x, np.max([s, np.zeros(len(s))], 0), label='EI')
      #ax[1].plot(x, np.max([s2, np.zeros(len(s))], 0), label='NEI', ls='--')
      #ax[0][1].plot(x, np.max([np.sqrt(np.array(s) ** 2 + np.array(s2) ** 2), np.zeros(len(s))], 0), label=r'$\sqrt{\sum_i EI(f_i)^2}$', ls='-.')
      #ax[1].twinx().plot(x, np.max([np.sqrt(np.array(s) ** 2 + np.array(s2) ** 2), np.zeros(len(s))], 0), label='NEI', ls='-.')
      if KNOWLEDGE:
         ax2 = ax[0][1].twinx()
         kngdnt = np.array([-1 * KG(xc) for xc in x])
         kngdnt2 = np.array([-1 * KG(xc, gpn=1) for xc in x])
         ax2.plot(x, kngdnt, label=r'$KG(f_1)$', c='purple')
         ax2.plot(x, kngdnt2, label=r'$KG(f_2)$', c='purple', ls='--')
         ax2.plot(x, np.sqrt(kngdnt ** 2 + kngdnt2 ** 2), label=r'$\sqrt{\sum_i EI(f_i)^2}$', ls='-.', c='purple')
      ax[1][0].scatter(b.T[:, 0], b.T[:, 1], c='blue', marker='s')
      ax[1][0].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)), s=25)
      ax[1][0].legend(loc='upper left')
      #s2 = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf)[0] for xc in spo]
      #ax[1].scatter([thisX], [(pointdic[' '.join((str(s) for s in thisX))])], c='red')
      #ax[1].scatter(spo, [KG(xc) for xc in spo], s=15, c='green', lw=3)
      #ax[1].scatter(spo, s2, s=15, c='green', lw=3)
      #ax[1].scatter(spo, np.max([np.zeros(len(s2)), s2], 0), s=15, c='green', lw=3)

      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False, truth=True)
      d = b#[:, c]
      ax[1][1].scatter(d.T[:, 0], d.T[:, 1], c='blue', marker='s')

      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False, knowgrad=False)
      d = b[:, c]
      ax[1][1].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))

      ax[0][0].set_title(r"%i High Fidelity Evaluations" % (NEVALS))
      plt.savefig('gp%05d' % __)
      plt.clf()
      plt.close('all')

   if False:
   #if KNOWLEDGE:
   #if __ < 2:
      pointdic[' '.join((str(s) for s in min_KG_x))] = f(min_KG_x)
      thisX = min_KG_x
      hey
   else:
      if ' '.join((str(s) for s in parX)) in pointdic.keys(): hey
      pointdic[' '.join((str(s) for s in parX))] = f(parX)
      thisX = parX
   NEVALS += 1
   outf.write(' '.join(
              [str(s) for s in thisX] + 
              [str(pointdic[' '.join((str(s) for s in thisX))])] + 
              ['\n']
              ))
      
   outf.close()

  # if min_val > -3e-7: break
   if __ > 8: break
   #if __ > 2 and min_val > -1e-5: break

keys = np.array([key for key in pointdic.keys()])
vals = [float(pointdic[v]) for v in keys]
x0 = keys[vals == np.min(vals)]
#x0 = [float(s) for s in keys[vals == np.min(vals)].split(' ')]
