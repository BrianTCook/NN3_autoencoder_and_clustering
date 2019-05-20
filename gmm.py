import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from astroML.plotting.tools import draw_ellipse

data = pd.read_csv('encoded_with_labels.txt')

data.columns= ['company', 'year', 'x', 'y']

company = data['company'].tolist()
year = data['year'].tolist()

l = [company[i] + ', ' + str(year[i]) for i in range(len(company))]
x = data['x'].tolist()
y = data['y'].tolist()

bigtech_companies = ['AMZN', 'GOOGL', 'AAPL', 'ORCL', 'MSFT', 'FB', 'IBM']
data_justtech = data.loc[data['company'].isin(bigtech_companies)]

company_bigtech = data_justtech['company'].tolist()
year_bigtech = data_justtech['year'].tolist()

l_bt = [company_bigtech[i] + ', ' + str(year_bigtech[i]) for i in range(len(company_bigtech))]
x_bt = data_justtech['x'].tolist()
y_bt = data_justtech['y'].tolist()

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

def plot_gmm(firstvar, secondvar, labels, name):
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
    
    plt.figure()
    
    plt.scatter(firstvar, secondvar, s=2)
    plt.xlabel(r'$x$', fontsize=20)
    plt.ylabel(r'$y$', fontsize=20)
    plt.title('Gaussian Mixture Model, %s'%(name), fontsize=14)
    plt.tight_layout()
    
    for mu, C, w in zip(gmm_best.means_, gmm_best.covariances_, gmm_best.weights_):
        draw_ellipse(mu, C, scales=[2], fc='none', ec='k')
        
    if name == 'bigtech':
        for i in range(len(firstvar)):
            plt.text(firstvar[i], secondvar[i], labels[i], color="red", fontsize=4)

    plt.savefig('gmm_%s.pdf'%name)

plot_gmm(x,y,l,'allstocks')
plot_gmm(x_bt,y_bt,l_bt,'bigtech')
