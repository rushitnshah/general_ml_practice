import argparse
import numpy as np
from rich import print
from sklearn import datasets
from sklearn.model_selection import train_test_split
'''
Logistic Regression


1. Math
    C classes
    N samples
    d dimensions
    labels = [0,1,2,3,...]

    W - weight matrix 
        one w for each class
        each w.shape = (d,)
        => W.shape = (d,C)

    p_c = P(y_pred = c | x) = exp(-(w_c.T).dot(x))/ sum_c'(exp(-(w_c.T).dot(x)))
    logits = (W.T).dot(x) # (C,)
    p = softmax()

    Gradient grad_W
        one for each w
        grad_w.shape: (d,)
        => grad_W.shape = (d,C)
        grad_wc = (p_c - y_c)*x \forall c
        grad_W = (p - y_onehot)*x

2. Structure
    - sklearn API
        model = LogisticRegression(lr=0.01,...)
        model.fit(X,Y)

    - Methods:
        a) fit
    
    - Utils:
        a) softmax
        b) cross entropy loss
'''
def softmax(logits, axis):
    """logits.shape = (C,B)"""
    exp_logits = np.exp(logits)
    return exp_logits/exp_logits.sum(axis=axis)

def crossentropy_loss(probs, y_onehot):
    """
    logits.shape = (C,B)
    y_onehot.shape = (C,B)
    crossent  = 1/N sum_N (sum_C -y_c*log(p_c))
    """
    return (-y_onehot*np.log(probs)).sum(axis=0).mean()

class DataGenerator:
    def __init__(self, X, Y, Y_oneh):
        self.X = X
        self.Y = Y
        self.Y_oneh = Y_oneh
        self.d, self.N = self.X.shape
        self.pos = 0
        self.inds = np.arange(self.N)
        np.random.shuffle(self.inds)
    
    def sample(self, batch_size=1):
        ind = np.random.choice(self.inds)
        if batch_size == 1:
            return np.expand_dims(self.X[:,ind],axis=1), self.Y[ind], np.expand_dims(self.Y_oneh[:,ind], axis=1)

        return self.X[:,ind], self.Y[ind], self.Y_oneh[:,ind]

class LogisticRegression:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def fit(self, X, Y, max_epochs=1001, nbatches=None, eval_freq=50):
        d, N = X.shape
        C = np.unique(Y).size
        Y_onehot = np.zeros((C,N))
        Y_onehot[Y, np.arange(Y.size)] = 1
        inds = np.arange(N)
        datagenerator = DataGenerator(X, Y, Y_onehot)
        

        W = np.zeros((d,C))

        for epoch in range(max_epochs):
            np.random.shuffle(inds)
            correct = 0
            loss = 0
            for i in range(N):
                # Sample an x, y
                # x, y, y_oneh = X[:,inds[i]], Y[inds[i]], Y_onehot[:,inds[i]]
                x, y, y_oneh = datagenerator.sample()

                # Predict
                logits = (W.T).dot(x)                                  # (C, B)
                probs = softmax(logits, axis=0)                        # (C, B)
                y_pred = np.argmax(probs, axis=0)

                # Loss/acc
                loss += crossentropy_loss(probs, y_oneh)
                correct += (y_pred == y).sum()

                # Grad update
                # grad_W = (probs - y_onehot)*x         # ((C,B) - (C,B))*(d,B)
                grad_W = x.dot((probs - y_oneh).T)                    # x (d,B)

                # Update W
                W -= self.lr*grad_W
            
            acc = correct / N
            mean_loss = loss  / N
            if epoch % eval_freq == 0:
                print(f" epoch: {epoch:4d}   |   loss {mean_loss:04.4f}  |   acc {acc:2.2f}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='iris')
    args = parser.parse_args()
    np.random.seed(42)
    
    # Prepare Data
    if args.data == 'iris':
        iris = datasets.load_iris()
        X = iris["data"]
        Y = iris["target"].astype(np.int8)
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.20, random_state=42)
    elif args.data == 'synthetic':
        C = 2
        N = 150
        Nc = int(N/C)
        d = 3
        X = np.zeros((N,d))
        Y = np.zeros((N,)).astype(np.int8)
        
        mus = [[-10,-10,-10], [10,10,10], [-10,10,-10]]
        cov = 5*np.eye(d)
        for i in range(C):
            X[i*Nc:Nc*(i+1),:] = np.random.multivariate_normal(mus[i],cov, (Nc,))
            Y[i*Nc:Nc*(i+1)] = i*np.ones(Nc)
        # mus = [-5, 10, 25]
        # for i in range(C):
        #     X[i*Nc:Nc*(i+1),:] = np.random.normal(mus[i],5, (Nc,d))
        #     Y[i*Nc:Nc*(i+1)] = i*np.ones(Nc)
        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.20, random_state=42)
        
    X_tr = X_tr.T
    Y_tr = Y_tr.astype(np.int8)
    X_te = X_te.T
    Y_te = Y_te.astype(np.int8)

    model = LogisticRegression()
    max_epochs = 10
    model.fit(X_tr, Y_tr, max_epochs=max_epochs, eval_freq=max(1, max_epochs//50))
