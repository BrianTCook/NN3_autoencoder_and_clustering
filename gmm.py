import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from astroML.plotting.tools import draw_ellipse

feh, m, d = np.loadtxt('gc.dat', unpack=True)
logd = np.log10(d)

firstvar, secondvar = feh, logd
X = np.vstack([firstvar, secondvar]).T # GaussianMixture requires a 2D array as input

K = np.arange(1, 6)
models = [None for i in K]

models = [GaussianMixture(K[i], random_state=1, covariance_type='full', \
          n_init=10).fit(X) for i in range(len(K))]

AIC = [m.aic(X) for m in models]
BIC = [m.bic(X) for m in models]
logL = [m.score(X) for m in models]
for i in range(len(K)):
    print('K = %d  AIC = %.1f  BIC = %.1f  logL = %.3f'%(K[i], AIC[i], BIC[i], logL[i]))

gmm_best = models[np.argmin(AIC)] # choose the best model with smallest AIC

print('mu =', gmm_best.means_.flatten())
print('sig =', np.sqrt(gmm_best.covariances_.flatten()))
print('pk =', gmm_best.weights_.flatten())

plt.scatter(firstvar, secondvar)
#plt.xlabel('[Fe/H]')
#plt.ylabel('log d (kpc)')
#plt.ylabel('m')
for mu, C, w in zip(gmm_best.means_, gmm_best.covariances_, gmm_best.weights_):
    draw_ellipse(mu, C, scales=[2], fc='none', ec='k')
plt.show()
