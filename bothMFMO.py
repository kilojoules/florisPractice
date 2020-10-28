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


def expected_improvement(X, X_sample, Y_sample, gpr, xi=.1):
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
    #print(X.shape)
    #print(gpr(X_sample)) 
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

    #print('MU IS ', mu, sigma)
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
       #print(a) 
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
   kernel = Matern(28.8, (1e-2 , 5e2 ), nu=1.5) 
   #kernel = Matern(np.ones(DIM) * 1, (1e-8 , 1e1 ), nu=1.4) #+ C(1e-2, (1e-8, 10))
   #kernel = RBF(np.ones(DIM) * 1e-2 , (1e-8 , 5e1 )) #+ WhiteKernel(1e-2)
   #kernel = RBF(np.ones(2) * 15.7 , (.3 , 5e2 )) #+ RBF(np.ones(DIM) * 1e-6, (1e-9, 1e-2))
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
   #parX = a[:, c][:, np.argmax((np.max(np.abs(b[:, c]), 0)))]
   #parX = a[:, c][:, np.argmin((np.max(np.abs(b[:, c]), 0)))]

   #parX = a[:, c][:, np.argmax(np.sqrt(np.sum(b[:, c] ** 2, 0)))]
   parX = a[:, c][:, np.argmin(np.sqrt(np.sum(b[:, c] ** 2, 0)))]
   val = np.min(np.max((np.sqrt(b[:, c] ** 2)), 0))
   #val = np.max((np.sqrt(np.sum(b[:, c] ** 2, 0))))
   maxval = np.max((np.sqrt(np.sum(b[:, c] ** 2, 0))))
   d = b[:, c]

   if True:
      print("PROBE")
      print(pointdic)
      fig, ax = plt.subplots(3, 4, figsize=(10, 10))
      # f1_true f1_ei    f2_true ..
      # f1_mean f1_std    ...    f2_std
      #       estimated PF    True PF
      plt.subplots_adjust(wspace=.3, hspace=0.5)
      x = np.linspace(XL, XU, 12)#[1:-1]
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



      c = ax[0][0].contourf(X, Y, fs1, 13)
      fig.colorbar(c, ax=ax[0][0])
      c = ax[1][0].contourf(X, Y, fs2, 13)
      fig.colorbar(c, ax=ax[1][0])
      ax[0][0].set_title('True HF $f_1$')
      ax[1][0].set_title('True HF $f_2$')
      

      c = ax[0][1].contourf(X, Y, gs1, 13)
      fig.colorbar(c, ax=ax[0][1])
      c = ax[1][1].contourf(X, Y, gs2, 13)
      fig.colorbar(c, ax=ax[1][1])
      ax[0][1].set_title('Approx $\mu(f_1)$')
      ax[1][1].set_title('Approx $\mu(f_2)$')

      c = ax[0][2].contourf(X, Y, gstd1, 13, cmap=plt.cm.coolwarm)
      fig.colorbar(c, ax=ax[0][2])
      c = ax[1][2].contourf(X, Y, gstd2, 13, cmap=plt.cm.coolwarm)
      fig.colorbar(c, ax=ax[1][2])
      ax[0][2].set_title('Approx $\sigma(f_1)$')
      ax[1][2].set_title('Approx $\sigma(f_2)$')
      c = ax[0][3].contourf(X, Y, -1 * ei1, 13)
      fig.colorbar(c, ax=ax[0][3])
      c = ax[1][3].contourf(X, Y, -1 * ei2, 13)
      fig.colorbar(c, ax=ax[1][3])
      ax[0][3].set_title('$EI(f_1)$')
      ax[1][3].set_title('$EI(f_2)$')
      #ax[0].clabel(c, inline=1, fontsize=9, fmt='%.2e')

      #c = ax[1][1].contour(X, Y, -1 * projectedKG, 13)
     

      xy = np.array([[float(k.split(' ')[0]) for k in keys], [float(k.split(' ')[1]) for k in keys]])
      for qq in range(2):
         for oo in range(4):
            ax[qq][oo].scatter(xy[0, :], xy[1, :], marker='*', s=15, lw=3)



      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False, truth=True)
      d = b[:, c]
      c = ax[2][0].scatter(d.T[:, 0], d.T[:, 1], c='red', marker='s')

      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=False)
      d = b[:, c]
      c = ax[2][0].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))
      cb = fig.colorbar(c, ax=ax[2][0])
      cb.set_label(r'$\sum_i f_i^j$')

      a, b, c = parEI(gpf1, gpf2, X_sample, Y_sample, EI=True)
      d = b[:, c]
      c = ax[2][3].scatter(d.T[:, 0], d.T[:, 1], c=np.sqrt((d.T[:, 0] ** 2 +  d.T[:, 1] ** 2)))
      cb = fig.colorbar(c, ax=ax[2][3])
      cb.set_label(r'$\sum_i EI(f_i)^j$')
      ax[2][0].set_xlabel(r'$f_1$')
      ax[2][0].set_ylabel(r'$f_2$')
      ax[2][3].set_xlabel(r'$-EI(f_1)$')
      ax[2][3].set_ylabel(r'$-EI(f_2)$')

      plt.suptitle(r"Minimize of $\sum_i EI_i^j$ %i High Fidelity Evaluations" % (NEVALS))
      #ax[0][1].set_title("Expected Improvement")
      #ax[1][0].set_xlabel(r'$EI(f_1)$')
      #ax[1][0].set_ylabel(r'$EI(f_2)$')
      #ax[1][1].set_xlabel(r'$f_1$')
      #ax[1][1].set_ylabel(r'$f_2$')
      ax[2][1].remove()
      ax[2][2].remove()
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
