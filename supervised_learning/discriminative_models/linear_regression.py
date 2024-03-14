import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X, Y, ltype='analytic', alpha=0.0, verbose=False):
    N, d = X.shape
    if ltype == 'analytic':
        """
        \theta = (X^T X)^-1 X^T Y
        where, 
        X.shape = N x d
        X.T.shape = d x N
        X.T.dot(X).shape = (d x N) x (N x d) = (dxd)
        Y.shape = N
        X.T.dot(Y).shape = (d x N) x (N) = (d)
        \theta.shape = (dxd) x d = d
        """
        A = np.eye(d)
        A[-1,-1] = 0    # turn o-ff regulatrization for bias term
        xtx_inv = np.linalg.inv(X.T.dot(X) + alpha*A)
        w = xtx_inv.dot(X.T.dot(Y))
        Y_pred = X.dot(w)
    elif ltype == 'batch_gd':
        w = np.random.rand(d)
        iters = 1000
        learning_rate = np.sqrt(1/iters)
        for iter in range(iters):
            grad_w = (2/N * X.T.dot(X.dot(w) - Y)) + 2*alpha*w
            w -= learning_rate*grad_w
            rmse = np.sqrt(((X.dot(w) - Y)**2).mean())
            if iter % 50 == 0 and verbose:
                print(f"iter {iter}: rmse {rmse:04.4f}")
        Y_pred = X.dot(w)
    
    elif ltype == 'sgd':
        def lr_scheduler(t0, t1):
            def lr_schedule(t):
                return t0 / (t + t1)
            return lr_schedule
        w = np.random.rand(d)
        nepochs = 1000
        lr_schedule = lr_scheduler(t0=5,t1=50)
        idxs = np.arange(N, dtype=np.int8)
        for epoch in range(nepochs):
            np.random.shuffle(idxs)
            for i in range(N):
                idx = idxs[i]
                grad_w = 2 * X[idx].T.dot(X[idx].dot(w) - Y[idx]) + 2*alpha*w
                lr = lr_schedule(t= epoch*N + i)
                w -= lr*grad_w
        rmse = np.sqrt(((X.dot(w) - Y)**2).mean())
        if epoch % 50 == 0 and verbose:
            print(f"epoch {iter}: rmse {rmse:04.4f}")
        Y_pred = X.dot(w)
    
    elif ltype == 'sklearn':
        from sklearn.linear_model import LinearRegression
        sklean_linregr_analytic = LinearRegression()
        sklean_linregr_analytic.fit(X, Y)
        w, b = sklean_linregr_analytic.coef_, sklean_linregr_analytic.intercept_
        Y_pred = sklean_linregr_analytic.predict(X)
    
    elif ltype == 'sklearn_sgd':
        from sklearn.linear_model import SGDRegressor
        sklean_linregr_sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.1, penalty=None)
        sklean_linregr_sgd.fit(X, Y)
        w, b = sklean_linregr_sgd.coef_, sklean_linregr_sgd.intercept_
        Y_pred = sklean_linregr_sgd.predict(X)
    
    rmse = np.sqrt(((Y_pred - Y)**2).mean())
    return w, rmse

if __name__=='__main__':
    np.random.seed(42)

    # Generate training data
    N = 100
    d = 8
    x = 2*np.random.rand(N,d)
    x = np.concatenate((x,np.ones((N,1))), axis=1)
    w_true = 5*np.random.rand(d+1)
    y_true = x.dot(w_true)

    # Linear regression parameters
    alpha = 0.5

    norm = lambda x: np.linalg.norm(x)
    
    # Solve analytically
    w_analytic,rmse_analytic = linear_regression(x,y_true, 'analytic', alpha=alpha)
    ccf_analytic = (w_true/norm(w_true)).dot(w_analytic/norm(w_analytic))
    
    # Solve batch grad descent
    w_batch_gd,rmse_batch_gd = linear_regression(x,y_true, 'batch_gd', alpha=alpha)
    ccf_batch_gd = (w_true/norm(w_true)).dot(w_batch_gd/norm(w_batch_gd))
    
    # Solve batch grad descent
    w_sgd, rmse_sgd = linear_regression(x,y_true, 'sgd', alpha=alpha)
    ccf_sgd = (w_true/norm(w_true)).dot(w_sgd/norm(w_sgd))
    
    # Solve using sklearn + analytic
    w_sklearn, rmse_sklearn = linear_regression(x,y_true, 'sklearn')
    ccf_sklearn = (w_true/norm(w_true)).dot(w_sklearn/norm(w_sklearn))
    
    # Solve using sklearn + sgd
    w_sklearn_sgd, rmse_sklearn_sgd = linear_regression(x,y_true, 'sklearn_sgd')
    ccf_sklearn_sgd = (w_true/norm(w_true)).dot(w_sklearn_sgd/norm(w_sklearn_sgd))

    print(f" ____________________________________________________________")
    print(f"|        Analytic         |   RMSE {rmse_analytic:04.4f}     |   ccf {ccf_analytic:02.4f}  |")
    print(f"|        Batch GD         |   RMSE {rmse_batch_gd:04.4f}     |   ccf {ccf_batch_gd:02.4f}  |")
    print(f"|     Stochastic GD       |   RMSE {rmse_sgd:04.4f}     |   ccf {ccf_sgd:02.4f}  |")
    print(f"|    sklearn Analytic     |   RMSE {rmse_sklearn:04.4f}     |   ccf {ccf_sklearn:02.4f}  |")
    print(f"|      sklearn SGD        |   RMSE {rmse_sklearn:04.4f}     |   ccf {ccf_sklearn:02.4f}  |")
    print(f" ____________________________________________________________")