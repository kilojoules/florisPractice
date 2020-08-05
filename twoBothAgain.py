import numpy as np
import matplotlib.pyplot as plt
from patternSearch import patternSearch as ps
from scipy.stats import norm
from twofuncBoth import f
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.interpolate import Rbf
from scipy.optimize import minimize as mini
from scipy.optimize import fmin_cobyla

def callb(x): print('--- >', x)


plt.style.use('dark_background')

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.05):
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
    mu_sample = np.array([gpr(xx) for xx in X_sample])

    sigma = sigma.reshape(-1, 1)

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.min(mu_sample)

    #print('mu is ', mu)
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei =  imp * norm.cdf(Z) - sigma * norm.pdf(Z)
        ei[sigma <= 1e-8] = 0.0

    #print("HEY!!!!! EI IS ", ei)
    #return (mu - 2 * sigma)[0]
    return ei[0]


pointdic = {}
DIM = 2
XI = 0.1
NEVALS = 0
n_initial = 3
XL, XU = (-30, 30)
FULLBAYES = True
KNOWLEDGE = True
bounds = (np.ones(DIM) * XL, np.ones(DIM) * XU)
np.random.seed(184)
initial_samps = [np.random.uniform(XL, XU, size=DIM) for _ in range(n_initial)]
outf = open('log.log', 'w')
outf.write('y1 y2 y3 y4 pow\n')

for point in initial_samps:
    pointdic[' '.join((str(s) for s in point))] = f(point)
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

   thesePoints = thesePoints
   #thesePoints = np.atleast_2d(thesePoints).T
   #kernel = Matern(np.ones(DIM) * 1, (1e-8 , 5e6 ), nu=1.5) + C(1e-2, (1e-8, 1e8))
   #kernel = Matern(np.ones(DIM) * 1, (1e-8 , 1e1 ), nu=1.4) #+ C(1e-2, (1e-8, 10))
   #kernel = RBF(np.ones(DIM) * 1e-2 , (1e-8 , 5e1 )) #+ WhiteKernel(1e-2)
   kernel = RBF(5 , (.3 , 5e2 )) #+ RBF(np.ones(DIM) * 1e-6, (1e-9, 1e-2))
   #kernel = RBF(np.ones(DIM) * 10 , (.3 , 15 )) *  C(1e-2, (1e-8, 1e8)) + C(0, (1e-8, 1e8)) + 
   #kernel = (RBF(np.ones(DIM) * 5 , (.3 , 300 )) + RBF(np.ones(DIM) * 5 , (1e-3 , 3))) * RationalQuadratic(10)
   #kernel = C(1e-6, (1e4, 1e8)) * (RBF(np.ones(DIM) * 5 , (.3 , 300 )) + RBF(np.ones(DIM) * 5 , (1e-3 , 3))) #* RationalQuadratic(.1)
   #kernel = C(.1, (1e2, 1e10)) * RBF(np.ones(DIM) * 5 , (.3 , 300 )) #* RationalQuadratic(.1)
   gp_alpha = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True)
   gp_alpha.fit(np.array(thesePoints), theseEvals)
   #gp_alpha.fit(2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals)
   #gp_alpha.fit(2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals - np.array([g(this, XI) for this in thesePoints]))
   def gpf(x, return_std=False):
      alph, astd = gp_alpha.predict(np.atleast_2d(x), return_std=True)
      #print('----> ', x)
      if return_std:
         return (alph, astd)
         #return (alph + g(x, XI), astd)
      else:
         return alph 
         #return alph + g(x, XI)


   X_sample, Y_sample = (np.array(thesePoints), theseEvals)
   #X_sample, Y_sample = (2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals)
   #X_sample, Y_sample = (2 * (np.array(thesePoints) - XL) / (XU - XL) - 1, theseEvals - np.array([g(this, XI) for this in thesePoints]))



   min_val = 1e100
   for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(20)]:
      #print(x0)
      #res = mini(gpf, x0=x0, bounds=[(0, 3) for ss in range(DIM)], method='Nelder-Mead')
      res = mini(expected_improvement, x0=x0, bounds=[(XL, XU) for ss in range(DIM)], args=(X_sample, Y_sample, gpf), callback=callb) 
      if res.fun < min_val:
         min_val = res.fun
         min_x = res.x
   #hey


   def KG(x):
      points = [s for s in list(thesePoints) + [np.array(x)]]
      evals = [s for s in list(theseEvals) + [gpf(x)[0]]]
      gpnxt = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
      if DIM == 1:
         gpnxt.fit(np.atleast_2d(points).T, evals)
         #gpnxt.fit(np.array(points).reshape(-1, 1), evals)
      else: 
         #print('POINTS ', points)
         #print('EVALS ', evals)
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
         m2 = np.array([gpf_next(xc)[0] for xc in inx])
         s2 = np.array([gpf_next(xc, return_std=True)[1] for xc in inx])[:, 0]
         s1 = np.array([gpf(xc, return_std=True)[1] for xc in inx])[:, 0]
         print(s1.shape, m1.shape)
         plt.fill_between(inx, m1 - 2 * s1, m1 + 2 * s1, facecolor='red', alpha=.2)
         plt.fill_between(inx, m2 - 2 * s2, m2 + 2 * s2, facecolor='blue', alpha=.2)
         plt.scatter(min_x, gpf(min_x), c='red')
         plt.scatter(min_next_x, gpf_next(min_next_x)[0], c='blue', marker='x')
         plt.savefig('hey/%.3f___%.5f.png' % (x, gpf(min_x)[0] - gpf_next(min_next_x)))

      return (expected_improvement(min_x, np.array(points), np.array(evals), gpf)[0] - expected_improvement(min_x, np.array(points), np.array(evals), gpf_next)[0])
      #return -1 * (gpf(min_x)[0] - gpf_next(min_next_x))[0]

   if KNOWLEDGE:
      min_KG_val = 1
      for x0 in [np.random.uniform(XL, XU, size=DIM) for oo in range(3)]:
         #print(x0)
         res = ps(KG, x0=x0, bounds=[(XL, XU) for ss in range(DIM)], deltaX=10)
         if res['f'] < min_KG_val:
            min_KG_val = res['f']
            min_KG_x = res['x']

   if True:
      print("PROBE")
      fig, ax = plt.subplots(2, 2, figsize=(12, 4))
      x = np.linspace(XL, XU, 52)[1:-1]
      keys = pointdic.keys()
      keys = [str(key) for key in keys]
      X, Y = np.meshgrid(x, x)
      fs = np.array([np.array([f([xc, yc]) for xc in x]) for yc in x])
      gs = np.array([np.array([gpf([xc, yc])[0] for xc in x]) for yc in x])
      gstd = np.array([np.array([gpf([xc, yc], return_std=True)[1][0] for xc in x]) for yc in x])
      #gs = np.array([np.array([gpf([xc, yc])[0] for xc in x]) for yc in x])
      ei = np.array([np.array([expected_improvement([xc, yc],  X_sample, Y_sample, gpf)[0] for xc in x]) for yc in x])
      #nei = np.array([np.array([expected_improvement([xc, yc],  X_sample, Y_sample, gpf_next)[0] for xc in x]) for yc in x])
      projectedKG = np.array([np.array([KG([xc, yc]) for xc in x]) for yc in x])
      #kngdnt = np.array([KG(xc) for xc in x])
      #kngdnt = np.array([KG([xc]) for xc in x])
      if not FULLBAYES: ax[0].plot(x, g([x], XI), label='Low Fidelity', c='blue')
      c = ax[0][0].contour(X, Y, fs, 13)
      #ax[0].clabel(c, inline=1, fontsize=9, fmt='%.2e')
      c = ax[0][1].contour(X, Y, gstd, 6, cmap=plt.cm.coolwarm)
      #ax[1].clabel(c, inline=1, fontsize=9, fmt='%.2e')
      c = ax[0][1].contour(X, Y, gs, 13)
      #ax[1].clabel(c, inline=1, fontsize=9, fmt='%.2e')
      c = ax[1][0].contour(X, Y, -1 * ei, 13)
      c = ax[1][1].contour(X, Y, -1 * projectedKG, 13)
      #ax[2].clabel(c, inline=1, fontsize=9, fmt='%.2e')
      xy = np.array([[float(k.split(' ')[0]) for k in keys], [float(k.split(' ')[1]) for k in keys]])
      ax[1][0].scatter(xy[0, :], xy[1, :], c=[pointdic[key] for key in keys], marker='*', s=15, lw=3)
      ax[1][1].scatter(xy[0, :], xy[1, :], c=[pointdic[key] for key in keys], marker='*', s=15, lw=3)
      ax[0][1].scatter(xy[0, :], xy[1, :], c=[pointdic[key] for key in keys], marker='*', s=15, lw=3)
      #if KNOWLEDGE:
      #   ax2 = ax[1].twinx()
      #   ax2.plot(x, -1 * kngdnt, label='KG', c='purple')
      #ax[1][0].legend(loc='upper left')
      #ax2.legend(loc='upper right')
      #s2 = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf)[0] for xc in spo]
      ax[0][0].set_title(r"%i High Fidelity Evaluations" % (NEVALS))
      plt.savefig('gpkgtwo%05d' % __)
      #plt.savefig('gpkg%05d' % __)
      plt.clf()
      plt.close('all')

   if True:
   #if KNOWLEDGE:
   #if __ < 2:
      pointdic[' '.join((str(s) for s in min_KG_x))] = f(min_KG_x)
      thisX = min_KG_x
   else:
      pointdic[' '.join((str(s) for s in min_x))] = f(min_x)
      thisX = min_x
   NEVALS += 1
   outf.write(' '.join(
              [str(s) for s in thisX] + 
              [str(pointdic[' '.join((str(s) for s in thisX))])] + 
              ['\n']
              ))
      
   outf.close()

  # if min_val > -3e-7: break
   if __ > 2 and min_val > -3e-4: break


keys = np.array([key for key in pointdic.keys()])
vals = [float(pointdic[v]) for v in keys]
x0 = keys[vals == np.min(vals)]
#x0 = [float(s) for s in keys[vals == np.min(vals)].split(' ')]
