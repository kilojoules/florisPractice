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

plt.style.use('dark_background')

def callb(x): print('--- >', x)


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
    #mu_sample = np.array([gpr(xx)[0] for xx in X_sample])

    sigma = sigma.reshape(-1, 1)

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.min(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei =  imp * norm.cdf(Z) - sigma * norm.pdf(Z)
        #print('norm is ',  norm.cdf(Z), Z)
        #print('imp is ', imp, mu, mu_sample_opt)
        ei[sigma <= 1e-8] = 0.0

    #print("HEY!!!!! EI IS ", ei)
    #return (mu - sigma * 2)[0]
    return  ei[0]


pointdic = {}
DIM = 2
XI = 0.1
NEVALS = 0
n_initial = 2
XL, XU = (-30, 30)
FULLBAYES = True
KNOWLEDGE = False
bounds = (np.ones(DIM) * XL, np.ones(DIM) * XU)
np.random.seed(82)
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

   thesePoints = np.atleast_2d(thesePoints)
   #kernel = Matern(np.ones(DIM) * 1, (1e-8 , 5e6 ), nu=1.5) + C(1e-2, (1e-8, 1e8))
   #kernel = Matern(np.ones(DIM) * 1, (1e-8 , 1e1 ), nu=1.4) #+ C(1e-2, (1e-8, 10))
   kernel = RBF(5 , (1e-2 , 1e2 )) #+ WhiteKernel(1e-2)
   #kernel = RBF(np.ones(DIM) * 5 , (1e-2 , 1e2 )) #+ WhiteKernel(1e-2)
   #kernel = RBF(np.ones(DIM) * 10 , (.3 , 5e3 )) #+ RBF(np.ones(DIM) * 1e-6, (1e-9, 1e-2))
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

   if True:
      nextpoints = list(thesePoints) + [np.array(min_x)]
      nextevals = theseEvals + list(gpf(min_x))
      gpnxt = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=35, random_state=98765, normalize_y=True)
      if DIM == 1:
         #print('POINTS ', points)
         #print('EVALS ', evals)
         gpnxt.fit(np.array(nextpoints).reshape(-1, 1), nextevals)
      else: 
         gpnxt.fit(nextpoints, nextevals)
   
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
         res = mini(expected_improvement, x0=x0, bounds=[(XL, XU) for ss in range(DIM)], args=(np.array(nextpoints), np.array(nextevals), gpf_next)) 
         #res = mini(gpf_next, x0=x0, bounds=[(0, 3) for ss in range(DIM)], args=(X_sample, Y_sample, gpf_next))
      #print('--> ', res.fun, res.fun[0] < min_next_val)
         if res.fun < min_next_val:
            min_next_val = res.fun
            min_next_x = res.x

   #if np.random.random() > .5:
   ###if __ < 2:
   #   beta = True
   #   thisX = min_x
   #else:
   #   beta = False
   #   thisX = min_next_x
   thisX = min_x

   if True:
      print("PROBE")
      print(pointdic)
      fig, ax = plt.subplots(1, 3, figsize=(12, 4))
      x = np.linspace(XL, XU, 52)[1:-1]
      keys = pointdic.keys()
      keys = [str(key) for key in keys]
      X, Y = np.meshgrid(x, x)
      fs = np.array([np.array([f([xc, yc]) for xc in x]) for yc in x])
      gs = np.array([np.array([gpf([xc, yc])[0] for xc in x]) for yc in x])
      gstd = np.array([np.array([gpf([xc, yc], return_std=True)[1][0] for xc in x]) for yc in x])
      gs = np.array([np.array([gpf([xc, yc])[0] for xc in x]) for yc in x])
      ei = np.array([np.array([expected_improvement([xc, yc],  X_sample, Y_sample, gpf)[0] for xc in x]) for yc in x])
      nei = np.array([np.array([expected_improvement([xc, yc],  X_sample, Y_sample, gpf_next)[0] for xc in x]) for yc in x])
      #gstd = np.array([np.array([gpf([xc, yc], return_std=True)[1] for xc in x]) for yc in x])
      #gs = np.array([gpf(np.ones(DIM) * xc)[0] for xc in x])
      #gstd2 = np.array([gpf_next(np.ones(DIM) * xc, return_std=True)[1][0] for xc in x])
      #ax[0].fill_between(x, gs - gstd, gs + gstd, facecolor='gray')
      #ax[0].fill_between(x, gs2 - 2 * gstd2, gs2 + 2 * gstd2, facecolor='purple', alpha=0.6)
      if not FULLBAYES: ax[0].plot(x, g([x], XI), label='Low Fidelity', c='blue')
      c = ax[0].contour(X, Y, fs, 13)
      #ax[0].clabel(c, inline=1, fontsize=9, fmt='%.2e')
      c = ax[1].contour(X, Y, gstd, 6, cmap=plt.cm.coolwarm)
      #ax[1].clabel(c, inline=1, fontsize=9, fmt='%.2e')
      c = ax[1].contour(X, Y, gs, 13)
      #ax[1].clabel(c, inline=1, fontsize=9, fmt='%.2e')
      c = ax[2].contour(X, Y, -1 * ei, 13)
      #ax[2].clabel(c, inline=1, fontsize=9, fmt='%.2e')
      #fig.colorbar(c)
      #ax[0].plot(x, [gpf(np.ones(DIM) * xc)[0] for xc in x], label='Prediction', c='red')
      #ax[0].plot(x, [gpf_next(np.ones(DIM) * xc)[0] for xc in x], label='Prediction', c='purple')
      #plt.plot(x, g(x) + [gpf(np.ones(DIM) * xc)[0] for xc in x])
      #ax[0].plot(x, [f(xc) for xc in x], c='yellow', lw=1, label='High Fidelity')
      #ax[0].set_xlim(XL, XU)
      #ax[0].set_xlim(XL, XU)
      xy = np.array([[float(k.split(' ')[0]) for k in keys], [float(k.split(' ')[1]) for k in keys]])
      ax[1].scatter(xy[0, :], xy[1, :], c=[pointdic[key] for key in keys], marker='*', s=15, lw=3)
      ax[2].scatter(xy[0, :], xy[1, :], c=[pointdic[key] for key in keys], marker='*', s=15, lw=3)
      #s = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf)[0] for xc in x]
      #ax[1].set_yscale('log')
      #ax[1].plot(x, np.max([s, np.zeros(len(s))], 0) )
      #spo = [float(k.split(' ')[0]) for k in keys]
      #ax[1].plot(x, s, label='EI')
      #ax[1].plot(x, np.max([s, np.zeros(len(s))], 0), label='EI')
      #kngdnt = [-1 * gpf_next(xc)[0] for xc in x]
      #ax2 = ax[1].twinx()
      #kngdnt = [-1 * expected_improvement(xc, nextpoints, nextevals, gpf_next)[0] for xc in x]
      #ax2.plot(x, kngdnt, label='NEI', ls='--', c='purple')
      #ax2.legend(loc='upper right')
      #ax[1].legend(loc='upper left')
      #s2 = [-1 * expected_improvement(xc, X_sample, Y_sample, gpf)[0] for xc in spo]
      #ax[1].scatter([thisX], -1 * expected_improvement(thisX, X_sample, Y_sample, gpf)[0], c='red', s=4)
     # if beta:
     #    ax[1].scatter([thisX], -1 * expected_improvement(thisX, X_sample, Y_sample, gpf)[0], c='red', s=4)
     # else:
     #    ax2.scatter([thisX], -1 * expected_improvement(thisX, nextpoints, nextevals, gpf_next)[0], c='blue', s=4)
      #ax[1].scatter(spo, [KG(xc) for xc in spo], s=15, c='green', lw=3)
      #ax[1].scatter(spo, s2, s=15, c='green', lw=3)
      #ax[1].scatter(spo, np.max([np.zeros(len(s2)), s2], 0), s=15, c='green', lw=3)
      if FULLBAYES:
         if __ > 0: ax[0].set_title(r"%i High Fidelity Evaluations" % (NEVALS))
         else: ax[0].set_title(r"%i High Fidelity Evaluations" % (NEVALS))
         plt.savefig('gpb%05d' % __)
      else:
         plt.title(r"$\xi=%.2f$, %i High Fidelity Evaluations" % (XI, NEVALS))
         plt.savefig('gpb_mf%05d' % __)
      plt.clf()
      plt.close('all')
      hey

   pointdic[' '.join((str(s) for s in thisX))] = f(thisX)
   NEVALS += 1
   outf.write(' '.join(
              [str(s) for s in thisX] + 
              [str(pointdic[' '.join((str(s) for s in thisX))])] + 
              ['\n']
              ))
      
   outf.close()

  # if min_val > -3e-7: break
   if __ > 2 and min_val > -1e-7: break


keys = np.array([key for key in pointdic.keys()])
vals = [float(pointdic[v]) for v in keys]
x0 = keys[vals == np.min(vals)]
#x0 = [float(s) for s in keys[vals == np.min(vals)].split(' ')]
