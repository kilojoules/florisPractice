from twofunc import f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


plt.style.use('dark_background')

np.random.seed(55555255)
samples = np.random.uniform(-30, 30, 3)
LFSamples = np.array(list(samples) + list(np.random.uniform(-30, 30, 5)))
lfsamps = np.array([f(samp, lf=True) for samp in samples])
lfsamps2 = np.array([f(samp, lf=True) for samp in LFSamples])
hfsamps = np.array([f(samp) for samp in samples])

kernel = RBF(15, (.3 , 5e2 )) #+ WhiteKernel(1, (1e-2, 1e2))
wkernel = RBF(15 , (.3 , 5e2 )) + C(1, (5e-2, 1e2))
lf_alpha = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=None)
lf_alpha.fit(np.atleast_2d(LFSamples).T, lfsamps2)
hf_alpha = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=None)
hf_alpha.fit(np.atleast_2d(samples).T, hfsamps)
delta_alpha = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=98765, normalize_y=True, optimizer=None)
delta_alpha.fit(np.atleast_2d(samples).T, hfsamps - lfsamps)

x = np.linspace(-30, 30, 100)
lf = lf_alpha.predict(np.atleast_2d(x).T, return_std=True)
hf = hf_alpha.predict(np.atleast_2d(x).T, return_std=True)
delta = delta_alpha.predict(np.atleast_2d(x).T, return_std=True)

if True:
   trueHF = np.array([f(xc) for xc in x])
   trueLF = np.array([f(xc, lf=True) for xc in x])
   fig, ax = plt.subplots(1, 3, figsize=(13 / 2., 4 / 2.))
   plt.subplots_adjust(wspace=.4)
   ax[0].fill_between(x, lf[0] - lf[1], lf[0] + lf[1], label=r'$\pm \sigma$')
   ax[0].plot(x, trueLF, c='r', label='Truth')
   ax[0].plot(x, lf[0], label=r'$\mu$', c='w')
   ax[0].legend()
   ax[1].fill_between(x, hf[0] - hf[1], hf[0] + hf[1])
   ax[1].plot(x, hf[0], c='w')
   ax[1].plot(x, trueHF, c='r')
   ax[2].fill_between(x, delta[0] - delta[1], delta[0] + delta[1])
   ax[2].plot(x, trueHF - trueLF, c='r')
   ax[2].plot(x, delta[0], c='w')
   ax[0].scatter(LFSamples, lfsamps2, c='blue', s=10)
   ax[1].scatter(samples, hfsamps, c='blue', s=10)
   ax[2].scatter(samples, hfsamps - lfsamps, c='blue', s=10)
   titles = ['Low-Fidelity', 'High-Fidelity', r'$\delta$']
   for ii in range(3): 
      ax[ii].set_xlabel('Yaw Offset (deg)')
      ax[ii].set_title(titles[ii])
   plt.savefig('hey')
   plt.clf()
   plt.close('all')

   muPred = lf[0] + delta[0]
   stdPred = np.sqrt(lf[1] ** 2 + delta[1] ** 2)

   mu = lf[0] + delta[0]
   std = np.sqrt(lf[1] ** 2 + delta[1] ** 2)
   sig = 1 / (
               (1 / std ** 2) + (1 / hf[1] ** 2)
               )
   mean = sig * (hf[0] / hf[1] ** 2 + mu / std ** 2)

   plt.fill_between(x, mu - std, mu + std, facecolor='w', alpha=0.8)
   plt.fill_between(x, hf[0] - hf[1], hf[0] + hf[1], facecolor='purple', alpha=0.8)
   plt.fill_between(x, mean - np.sqrt(sig), mean + np.sqrt(sig), facecolor='orange', alpha=0.4)
   plt.plot(x, muPred, c='w', label='LF prediction')
   plt.plot(x, hf[0], label='HF prediction', c='purple')
   plt.plot(x, trueHF, label='Truth', c='r')


   plt.plot(x, mean, c='orange', label='Fused prediction')
   plt.scatter(samples, hfsamps, s=10, c='blue')
   plt.legend()
   plt.savefig('deltaInAction')
   plt.clf()
   plt.close('all')


