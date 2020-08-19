from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from patternSearch import patternSearch as ps
from scipy.stats import norm
from twofuncBoth import f as turbF, g
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
   return (np.array([turbF(x, lf=False), g(x, lf=False)]) 
          - np.array([turbF(x, lf=True), g(x, lf=True)]))

def f(x, lf=False):
   return [turbF(x, lf=lf), g(x, lf=lf)]

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
    print(X.shape)
    print(gpr(X_sample)) 
    mu, sigma = gpr(X, return_std=True)
    mu_sample = gpr(X_sample)
    #mu_sample = np.array([gpr(xx)[0] for xx in X_sample.T])

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
DIM = 2
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
    y = np.linspace(XL, XU, 50)
    x = np.array([y, y])
    ins = np.array([s for s in permutations(y, 2)]).T
    #ins = np.atleast_2d(x).T
    #evs = gp.predict(ins)
    if EI:
       eis = expected_improvement(ins, X_sample, Y_sample[:, 0], gpf1)
       eis2 = expected_improvement(ins, X_sample, Y_sample[:, 1], gpf2)
       pars = is_pareto_efficient_simple(np.array([eis, eis2]).T)
       return (ins, np.array([eis, eis2]), pars)
    else:
       if not truth:
          a = [gpf1(np.atleast_2d(xc).T)[0] for xc in ins.T]
              # gpf1(np.atleast_2d(ins.T))
          b = [gpf2(np.atleast_2d(([xc])).T)[0] for xc in ins.T]
          #b = gpf2(np.atleast_2d(ins.T))
       else: 
          a = [turbF(i) for i in ins.T]
          b = [g(i) for i in ins.T]
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
      #thesePoints.append(float(point))
      thesePoints.append(np.array([float(s) for s in point.split(' ')]))
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
   gp1.fit((thesePoints).T, np.array(theseEvals)[:, 0])
   gp2.fit((thesePoints).T, np.array(theseEvals)[:, 1])
   #gp1.fit((thesePoints), (np.array(theseEvals)[:, 0] - np.mean(np.array(theseEvals)[:, 0])) / np.std(np.array(theseEvals)[:, 0]))
   #gp1.fit((thesePoints), (np.array(theseEvals)[:, 1] - np.mean(np.array(theseEvals)[:, 1])) / np.std(np.array(theseEvals)[:, 1]))
   #gp_alpha.fit(np.array(thesePoints), theseEvals)
   #gp_alpha.fit(2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals)
   #gp_alpha.fit(2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals - np.array([g(this, XI) for this in thesePoints]))
   def gpf1(x, return_std=False):
      alph, astd = gp1.predict(np.atleast_2d(x).T, return_std=True)
      if len(x.shape) > 1:
         alph += np.array([f(xc, lf=True)[0] for xc in x.T])
      else:
         alph += f(x, lf=True)[0]
      if return_std:
         return (alph, astd)
      else:
         return alph 

   def gpf2(x, return_std=False):
      alph, astd = gp2.predict(np.atleast_2d(x).T, return_std=True)
      if len(x.shape) > 1:
         alph += np.array([f(xc, lf=True)[1] for xc in x.T])
      else:
         alph += f(x, lf=True)[1]
      if return_std:
         return (alph, astd)
      else:
         return alph 


   X_sample, Y_sample = (np.array(thesePoints), np.array(theseEvals))
   #X_sample, Y_sample = (2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals)
   #X_sample, Y_sample = (2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals - np.array([g(this, XI) for this in thesePoints]))



   a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample)
   parX = a[:, c][:, np.argmin(np.sqrt(np.sum(b[:, c] ** 2, 0)))]
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
      fig, ax = plt.subplots(3, 4, figsize=(8, 8))
      # f1_true f1_ei    f2_true ..
      # f1_mean f1_std    ...    f2_std
      #       estimated PF    True PF
      plt.subplots_adjust(wspace=.3)
      x = np.linspace(XL, XU, 12)[1:-1]
      y = np.array([x, x]).T
      X, Y = np.meshgrid(x, x)
      keys = pointdic.keys()
      keys = [str(key) for key in keys]

      fs1 = np.array([np.array([f([xc, yc])[0] for xc in x]) for yc in x])
      gs1 = np.array([np.array([gpf1(np.atleast_2d([xc, yc]).T)[0] for xc in x]) for yc in x])
      gstd1 = np.array([np.array([gpf1(np.atleast_2d([xc, yc]).T, return_std=True)[1][0] for xc in x]) for yc in x])
      ei1 = np.array([np.array([expected_improvement(np.atleast_2d([xc, yc]).T,  X_sample, Y_sample, gpf1)[0] for xc in x]) for yc in x])

      fs2 = np.array([np.array([f([xc, yc])[1] for xc in x]) for yc in x])
      gs2 = np.array([np.array([gpf2(np.atleast_2d([xc, yc]).T)[0] for xc in x]) for yc in x])
      gstd2 = np.array([np.array([gpf2(np.atleast_2d([xc, yc]).T, return_std=True)[1][0] for xc in x]) for yc in x])
      ei2 = np.array([np.array([expected_improvement(np.atleast_2d([xc, yc]).T,  X_sample, Y_sample, gpf2)[0] for xc in x]) for yc in x])


      #gs1 = np.array([f((xc), lf=True)[0] + gpf1(np.atleast_2d(xc).T) for xc in x])[:, 0]
      #gs2 = np.array([f((xc), lf=True)[1] + gpf2(np.atleast_2d(xc).T) for xc in x])[:, 0]
      #gs2 = np.array([gpf_next(np.ones(DIM) * xc) for xc in x])
      #gstd1 = np.array([gpf1(np.atleast_2d(xc).T, return_std=True)[1] for xc in x])[:, 0]
      #gstd2 = np.array([gpf2(np.atleast_2d(xc).T, return_std=True)[1] for xc in x])[:, 0]
      #gstd2 = np.array([gpf_next(np.ones(DIM) * xc, return_std=True)[1] for xc in x])



      c = ax[0][0].contour(X, Y, fs1, 13)
      c = ax[1][0].contour(X, Y, fs2, 13)
      ax[0][0].set_title('True HF $f_1$')
      ax[1][0].set_title('True HF $f_2$')

      c = ax[0][1].contour(X, Y, gs1, 13)
      c = ax[1][1].contour(X, Y, gs2, 13)

      c = ax[0][2].contour(X, Y, gstd1, 6, cmap=plt.cm.coolwarm)
      c = ax[1][2].contour(X, Y, gstd2, 6, cmap=plt.cm.coolwarm)
      c = ax[0][3].contour(X, Y, -1 * ei1, 13)
      c = ax[1][3].contour(X, Y, -1 * ei2, 13)
      #ax[0].clabel(c, inline=1, fontsize=9, fmt='%.2e')

      #c = ax[1][1].contour(X, Y, -1 * projectedKG, 13)
     

      xy = np.array([[float(k.split(' ')[0]) for k in keys], [float(k.split(' ')[1]) for k in keys]])
      for qq in range(2):
         for oo in range(4):
            ax[qq][oo].scatter(xy[0, :], xy[1, :], marker='*', s=15, lw=3)



      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False, truth=True)
      d = b[:, c]
      ax[2][1].scatter(d.T[:, 0], d.T[:, 1], c='red', marker='s')

      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False)
      d = b[:, c]
      ax[2][1].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))

      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=True)
      d = b[:, c]
      ax[2][2].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))

      plt.suptitle(r"%i High Fidelity Evaluations" % (NEVALS))
      #ax[0][1].set_title("Expected Improvement")
      #ax[1][0].set_xlabel(r'$EI(f_1)$')
      #ax[1][0].set_ylabel(r'$EI(f_2)$')
      #ax[1][1].set_xlabel(r'$f_1$')
      #ax[1][1].set_ylabel(r'$f_2$')
      plt.savefig('gpMFMO%05d' % __)
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
