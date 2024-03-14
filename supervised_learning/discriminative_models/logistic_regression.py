import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from rich import print
from tqdm import tqdm

def sigmoid(w,x):
    return 1/(1+np.exp(-x.dot(w)))

def softmax(logits, axis):
    # logits.shape = B x C
    exp_scores = np.exp(logits) # B x C
    Z = exp_scores.sum(axis=axis)   # B
    probs = np.divide(exp_scores,np.expand_dims(Z,axis=1))
    return probs


def binary_cross_entropy(p,y):
    return - y*(np.log(p)) - (1-y)*np.log(1-p)

def crossentropy_loss(p, y_onehot):
    # p = Bxc
    # y = B
    # y_onehot = BxC
    return -(y_onehot*np.log(p)).sum(axis=1).mean()

class BatchGenerator:
    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.N, _ = self.X.shape
        self.inds = np.arange(self.N)
        self.reset()
    
    def reset(self):
        np.random.shuffle(self.inds)
        self.pos = 0
    
    def sample(self):
        p1 = self.pos   # 99
        p2 = p1 + self.batch_size # 100
        if p2 >= self.N:
            p2 = self.N # 100
            self.reset()
        self.pos += 1
        return self.X[p1:p2], self.Y[p1:p2] # 0 -> 255

def binary_logistic_regression(X,Y,verbose=False, ltype='sgd'):
    N, d = X.shape

    if ltype == 'sgd':
        batch_size = 1
    elif ltype == 'bgd':
        batch_size = N
    iters = int(N/batch_size)
    print(f"Iters : {iters}")
    
    data_generator = BatchGenerator(X, Y, batch_size=batch_size)
    
    w = np.zeros(d)
    max_epochs = 1000
    inds = np.arange(N)
    learning_rate = 0.001
    for epoch in range(max_epochs):
        np.random.shuffle(inds)
        correct = 0
        loss = 0
        for i in range(iters):
            idx  = inds[i]
            # x, y_true = X[idx], Y[idx]
            x, y_true = data_generator.sample()
            p = sigmoid(w,x)
            y_pred = (p >= 0.5).astype(np.int8)
            correct += (y_true == y_pred).astype(np.int8).sum()
            loss += binary_cross_entropy(p,y_true).mean()
            grad_w = (np.expand_dims(p - y_true, axis=1)*x).mean(axis=0) 
            w -= learning_rate*grad_w
        
        acc = correct/N
        mean_bce_loss = loss/N
        if epoch % 100 == 0 and verbose:
            print(f"Epoch {epoch} | loss {mean_bce_loss:2.4f} | acc {acc:2.2f}")

    return w

def logistic_regression(X,Y, ltype='sgd', verbose=True):
    N, d = X.shape
    num_classes = np.unique(Y).size
    W = np.zeros((d, num_classes))

    if ltype == 'sgd':
        batch_size = 1
    elif ltype == 'bgd':
        batch_size = N
    iters = int(N/batch_size)

    data_generator = BatchGenerator(X, Y, batch_size=batch_size)
    print(f"Iters : {iters}")

    max_epochs = 1000
    learning_rate = 0.01

    for epoch in range(max_epochs):
        correct = 0
        loss = 0
        for iter in range(iters):
            # sample
            x, y_true = data_generator.sample()
            y_onehot = np.zeros((y_true.size, num_classes)).astype(np.int8)
            y_onehot[np.arange(y_true.size),y_true] = 1

            # predict
            # x.shape - Bxd 
            # W.shape - dxC
            logits = x.dot(W) # B x C
            probs = softmax(logits, axis=1) # B x C
            y_pred = np.argmax(probs, axis=1).astype(np.int8) # B

            # loss/acc
            correct += (y_pred == y_true).astype(np.int8).sum()
            loss += crossentropy_loss(probs, y_onehot)

            # gradient + update
            grad_W = x.T.dot((probs - y_onehot))
            W -= learning_rate*grad_W
        
        acc = correct/N
        mean_ce_loss = loss/N
        if epoch % 100 == 0 and verbose:
            print(f"Epoch {epoch} | loss {mean_ce_loss:2.4f} | acc {acc:2.2f}")
    
    return W

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='binary', required=True)
    args = parser.parse_args()
    
    
    np.random.seed(42)
    
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris["data"]
    if args.type == 'binary':
        Y = (iris["target"] == 2).astype(np.int8)    # 1 if iris-virginica else 0
    else:
        Y = iris["target"].astype(np.int8)
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.20)#, random_state=42)

    if args.type == 'binary':
        # Stochastic gradient descent
        w = binary_logistic_regression(X_tr, Y_tr, ltype='sgd', verbose=True)

        # Full batch gradient descent
        w = binary_logistic_regression(X_tr, Y_tr, ltype='bgd', verbose=True)
    elif args.type == 'multiclass':
        # Stochastic gradient descent
        W = logistic_regression(X_tr, Y_tr, ltype='sgd')
        # Full batch gradient descent
        W = logistic_regression(X_tr, Y_tr, ltype='bgd')
    
    elif args.type == 'sklearn':
        from sklearn.linear_model import LogisticRegression
        logi_sklearn = LogisticRegression()
        logi_sklearn.fit(X_tr, Y_tr)
        Y_pred = logi_sklearn.predict(X_tr)
        acc = (Y_tr == Y_pred).sum()/X_tr.shape[0]
        print(f"Acc: {acc}")


