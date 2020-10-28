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

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def delta(x):
   return (np.array([turbF(x, lf=False)[0], g(x, lf=False)]) 
          - np.array([turbF(x, lf=True)[0], g(x, lf=True)]))

def f(x, lf=False):
   return [turbF(x, lf=lf)[0], g(x, lf=lf)]

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


plt.style.use('dark_background')


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
    #mu, sigma = gpr.predict(X, return_std=True)
    #mu_sample = gpr.predict(X_sample)
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

    print('MU IS ', mu, sigma)
    #print("HEY!!!!! EI IS ", ei)
    #print(mu - 2 * sigma)
    return np.min([ei, np.zeros(ei.shape)], 0)



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

for __ in range(120):

   outf = open('log.log', 'a')

   thesePoints, theseEvals = [], []
   for point in pointdic.keys():
      thesePoints.append(float(point))
      #thesePoints.append(np.array([float(s) for s in point.split(' ')]))
      theseEvals.append(pointdic[point])

   thesePoints = np.atleast_2d(thesePoints).T
   #kernel = Matern(np.ones(DIM) * 1, (1e-8 , 5e6 ), nu=1.5) + C(1e-2, (1e-8, 1e8))
   #kernel = Matern(np.ones(DIM) * 1, (1e-8 , 1e1 ), nu=1.4) #+ C(1e-2, (1e-8, 10))
   #kernel = RBF(np.ones(DIM) * 1e-2 , (1e-8 , 5e1 )) #+ WhiteKernel(1e-2)
   kernel = RBF(15.7 , (.3 , 5e2 )) #+ RBF(np.ones(DIM) * 1e-6, (1e-9, 1e-2))
   #kernel = RBF(np.ones(DIM) * 10 , (.3 , 15 )) *  C(1e-2, (1e-8, 1e8)) + C(0, (1e-8, 1e8)) + 
   #kernel = (RBF(np.ones(DIM) * 5 , (.3 , 300 )) + RBF(np.ones(DIM) * 5 , (1e-3 , 3))) * RationalQuadratic(10)
   #kernel = C(1e-6, (1e4, 1e8)) * (RBF(np.ones(DIM) * 5 , (.3 , 300 )) + RBF(np.ones(DIM) * 5 , (1e-3 , 3))) #* RationalQuadratic(.1)
   #kernel = C(.1, (1e2, 1e10)) * RBF(np.ones(DIM) * 5 , (.3 , 300 )) #* RationalQuadratic(.1)
   gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=None)
   gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=None)
   gp1.fit((thesePoints), np.array(theseEvals)[:, 0])
   gp2.fit((thesePoints), np.array(theseEvals)[:, 1])
   #gp1.fit((thesePoints), (np.array(theseEvals)[:, 0] - np.mean(np.array(theseEvals)[:, 0])) / np.std(np.array(theseEvals)[:, 0]))
   #gp1.fit((thesePoints), (np.array(theseEvals)[:, 1] - np.mean(np.array(theseEvals)[:, 1])) / np.std(np.array(theseEvals)[:, 1]))
   #gp_alpha.fit(np.array(thesePoints), theseEvals)
   #gp_alpha.fit(2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals)
   #gp_alpha.fit(2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals - np.array([g(this, XI) for this in thesePoints]))
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
   #X_sample, Y_sample = (2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals)
   #X_sample, Y_sample = (2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals - np.array([g(this, XI) for this in thesePoints]))



   a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample)
   parX = np.array([a[c][np.argmax(np.sqrt(np.sum(b[:, c] ** 2, 0)))]])
   #parX = np.array([a[c][np.argmin(np.sqrt(np.sum(b[:, c] ** 2, 0)))]])
   val = np.min((np.sqrt(np.sum(b[:, c] ** 2, 0))))
   maxval = np.max((np.sqrt(np.sum(b[:, c] ** 2, 0))))
   d = b[:, c]

   if False:
      min_val = 1e100
      for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(20)]:
         #print(x0)
         #res = mini(gpf, x0=x0, bounds=[(0, 3) for ss in range(DIM)], method='Nelder-Mead')
         res = mini(expected_improvement, x0=x0[0], bounds=[(XL, XU) for ss in range(DIM)], args=(X_sample, Y_sample, gpf), callback=callb) 
         if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
      #hey
 

      points = list(thesePoints) + [np.array(min_x)]
      evals = theseEvals + [gpf(min_x)[0]]
      gpnxt = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
      if DIM == 1:
         #print('POINTS ', points)
         #print('EVALS ', evals)
         gpnxt.fit(np.array(points).reshape(-1, 1), evals)
      else: 
         gpnxt.fit(points, evals)

      def gpf_next(x, return_std=False):
         alph, astd = gpnxt.predict(np.atleast_2d(x), return_std=True)
         alph = alph[0]
         if return_std:
            return (alph, astd)
         else:
            return alph

      min_next_val = 1
      for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(10)]:
         #res = mini(gpf_next, x0=x0, bounds=[(0, 3) for ss in range(DIM)])
         res = mini(expected_improvement, x0=x0, bounds=[(XL, XU) for ss in range(DIM)], args=(np.array(points), np.array(evals), gpf_next)) 
         #res = mini(gpf_next, x0=x0, bounds=[(0, 3) for ss in range(DIM)], args=(X_sample, Y_sample, gpf_next))
         #print('--> ', res.fun, res.fun[0] < min_next_val)
         if res.fun < min_next_val:
            min_next_val = res.fun
            min_next_x = res.x

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
      fig, ax = plt.subplots(2, 2, figsize=(8, 8))
      plt.subplots_adjust(wspace=.3)
      x = np.linspace(XL, XU, 302)[1:-1]
      keys = pointdic.keys()
      keys = [str(key) for key in keys]
      gs1 = np.array([f(np.array([xc]), lf=True)[0] + gpf1(np.ones(DIM) * xc) for xc in x])[:, 0]
      gs2 = np.array([f(np.array([xc]), lf=True)[1] + gpf2(np.ones(DIM) * xc) for xc in x])[:, 0]
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
      ax[0][0].plot(x, [turbF(xc) for xc in x], c='yellow', lw=1, label='High Fidelity')
      #axx.plot(x, [turbF(xc) for xc in x], c='yellow', lw=1, label='High Fidelity')
      axx.plot(x, [g(xc) for xc in x], c='yellow', lw=1, label='High Fidelity', ls='--')
      ax[0][0].set_xlim(XL, XU)
      keypoints = np.array([float(k.split(' ')[0]) for k in keys])
      ax[0][0].scatter(keypoints, np.array([f(np.array([point]), lf=True)[0] for point in keypoints]) + np.array([pointdic[key][0] for key in keys]), marker='*', s=15, c='green', lw=3)
      #ax[0][0].scatter([float(k.split(' ')[0]) for k in keys], [pointdic[key][0] for key in keys], marker='*', s=15, c='green', lw=3)
      #axx.scatter([float(k.split(' ')[0]) for k in keys], [pointdic[key][1] for key in keys], marker='*', s=15, c='lightgreen', lw=3, ls='--')

      s = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf1)[0] for xc in x]
      s2 = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf2)[0] for xc in x]
      #ax[1].plot(x, np.max([s, np.zeros(len(s))], 0) )
      spo = [float(k.split(' ')[0]) for k in keys]
      ax[0][1].plot(x, s, label=r'$EI(f_1)$')
      ax[0][1].twinx().plot(x, s2, label=r'$EI(f_2)$', ls='--')
      #ax[1].plot(x, np.max([s, np.zeros(len(s))], 0), label='EI')
      #ax[1].plot(x, np.max([s2, np.zeros(len(s))], 0), label='NEI', ls='--')
      ax[0][1].plot(x, np.max([np.sqrt(np.array(s) ** 2 + np.array(s2) ** 2), np.zeros(len(s))], 0), label=r'$\sqrt{\sum_i EI(f_i)^2}$', ls='-.')
      ax[0][1].legend(loc='upper right')
      #ax[1].twinx().plot(x, np.max([np.sqrt(np.array(s) ** 2 + np.array(s2) ** 2), np.zeros(len(s))], 0), label='NEI', ls='-.')
      #if KNOWLEDGE:
      #   ax2 = ax[1].twinx()
      #   kngdnt = [KG(xc) for xc in x]
      #   ax2.plot(x, kngdnt, label='KG', ls='--', c='purple')
      ax[1][0].scatter(b.T[:, 0], b.T[:, 1], c='blue', marker='s')
      cc = ax[1][0].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)), s=55)
      cbar = fig.colorbar(cc, ax=ax[1][0], format=ticker.FuncFormatter(fmt), orientation='horizontal')
      cbar.set_label(r'$\sqrt{\sum_i EI(f_i)^2}$')
      cbar.ax.set_yticklabels(cbar.ax.get_xticklabels(), rotation='vertical')
      ax[1][0].legend(loc='upper left')
      #s2 = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf)[0] for xc in spo]
      #ax[1].scatter([thisX], [(pointdic[' '.join((str(s) for s in thisX))])], c='red')
      #ax[1].scatter(spo, [KG(xc) for xc in spo], s=15, c='green', lw=3)
      #ax[1].scatter(spo, s2, s=15, c='green', lw=3)
      #ax[1].scatter(spo, np.max([np.zeros(len(s2)), s2], 0), s=15, c='green', lw=3)

      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False, truth=True)
      d = b#[:, c]
      ax[1][1].scatter(d.T[:, 0], d.T[:, 1], c='red', marker='s')

      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False)
      d = b[:, c]
      ax[1][1].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))

      plt.suptitle(r"%i High Fidelity Evaluations" % (NEVALS))
      ax[0][1].set_title("Expected Improvement")
      ax[1][0].set_xlabel(r'$EI(f_1)$')
      ax[1][0].set_ylabel(r'$EI(f_2)$')
      ax[1][1].set_xlabel(r'$f_1$')
      ax[1][1].set_ylabel(r'$f_2$')
      plt.savefig('gpMO%05d' % __)
      plt.clf()
      plt.close('all')

   if False:
   #if KNOWLEDGE:
   #if __ < 2:
      pointdic[' '.join((str(s) for s in min_KG_x))] = delta(min_KG_x)
      thisX = min_KG_x
      hey
   else:
      pointdic[' '.join((str(s) for s in parX))] = delta(parX)
      thisX = parX
   NEVALS += 1
   outf.write(' '.join(
              [str(s) for s in thisX] + 
              [str(pointdic[' '.join((str(s) for s in thisX))])] + 
              ['\n']
              ))
      
   outf.close()

   if __ > 1 and maxval < 1e-3: break
   if __ > 8: break
   #if __ > 2 and min_val > -1e-5: break

keys = np.array([key for key in pointdic.keys()])
vals = [float(pointdic[v]) for v in keys]
x0 = keys[vals == np.min(vals)]
#x0 = [float(s) for s in keys[vals == np.min(vals)].split(' ')]
