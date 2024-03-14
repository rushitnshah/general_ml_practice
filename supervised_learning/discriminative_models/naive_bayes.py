import math
import numpy as np
import scipy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from rich import print

class MultivariateNormalDistribution:
    """(1/sqrt(2*std))*exp(-0.5(x-mu)^2/var)
    """
    def __init__(self, loc, scale):
        self.K = loc.size
        self.mu = loc
        scale = np.maximum(scale, 1e-6*np.ones(self.K))
        self.cov = (scale**2)*np.eye(self.K)

        self.const_prefix = 1/np.sqrt(((2*math.pi)**self.K)*np.linalg.det(self.cov))
        self.cov_inv = np.linalg.inv(self.cov)
        # self.distribution = lambda x: (1/(scale*np.sqrt(2*math.pi)))*np.exp(-0.5*((x - loc)/scale)**2)

    def pdf(self, x):
        diff = x - self.mu
        prob  = self.const_prefix*np.exp(-0.5*(diff.T.dot(self.cov_inv)).dot(diff))
        return prob
        # return self.distribution(x)

class NaiveBayes:
    def __init__(self, X, Y):
        self.N, self.d = X.shape
        self.C = np.unique(Y)

        # Calculate priors, mean, and std for each class
        self.class_stats = {}
        for c in self.C:
            inds = np.where(Y==c)[0]
            x, y = X[inds], Y[inds]
            assert np.all(y == c)
            mu_c = np.mean(x, axis=0)
            std_c = np.std(x, axis=0)
            prior_prob = y.size/self.N
            self.class_stats[c] = {
                # 'distribution': scipy.stats.norm(mu_c, std_c),
                'distribution': MultivariateNormalDistribution(mu_c, std_c),
                'prior': prior_prob
            }
    
    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.predict_(x))
        return np.array(Y)
    
    def predict_(self, x):
        max_prob = float('-Inf')
        for c in self.C:
            likelihood = self.class_stats[c]['distribution'].pdf(x)
            prior = self.class_stats[c]['prior']
            posterior = likelihood*prior
            if posterior > max_prob:
                max_prob = posterior
                y_pred = c
        return y_pred



if __name__=='__main__':    
    np.random.seed(42)
    
    # Load Iris dataset
    iris = datasets.load_iris()
    noise = 0.0
    X = iris["data"] + np.random.normal(0, noise, X.shape)
    Y = iris["target"].astype(np.int8)

    X  += np.random.normal(0, 1.0, X.shape)

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.20)#, random_state=42)

    model_nb = NaiveBayes(X_tr, Y_tr)
    Y_pred = model_nb.predict(X_te)
    correct = (Y_pred == Y_te).sum()
    acc = correct/Y_te.size
    print(f"Test accuracy: {acc:2.2f}")

