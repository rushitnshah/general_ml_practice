import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from rich import print
from tqdm import tqdm

colors_members = ['pink','lightsteelblue','lime']
colors_means = ['r','g','b']
def plot(X, Y, kMeans, iter):
    _, ax = plt.subplots()
    grps = np.unique(Y)

    lgnd_entries = []
    for grp_id in grps:
        X_grp = X[np.where(Y==grp_id)[0],:]
        ax.scatter(X_grp[:, 0], X_grp[:, 1], c=colors_members[grp_id], marker='o')
        ax.scatter(kMeans[grp_id, 0], kMeans[grp_id, 1], c=colors_means[grp_id], marker='d')

        lgnd_entries.append(f"Cluster {grp_id}")
    
    plt.legend(lgnd_entries)

    ax.set(xlabel='x1', ylabel='x2')


    ax.set_title(f"Iteration {iter}")
    plt.savefig(f'./figs/kmeans_iter_{iter:03d}')
    plt.close()

if __name__=='__main__':
    np.random.seed(42)

    # Prepare data
    mu1 = np.array([-1,-2])
    mu2 = np.array([ 3, 4])
    mu3 = np.array([-1, 2])
    var = np.array([1.5, 1.5])

    x1 = np.random.normal(mu1, var, (100,2))
    y1 = np.zeros(100, dtype=np.int8)
    x2 = np.random.normal(mu2, var, (100,2))
    y2 = 1*np.ones(100, dtype=np.int8)
    x3 = np.random.normal(mu3, var, (100,2))
    y3 = 2*np.ones(100, dtype=np.int8)

    X = np.concatenate((x1,x2,x3), axis=0)
    Y = np.concatenate((y1,y2,y3))

    # k-Means Clustering
    k = 3
    d = X.shape[1]
    kMeans = np.random.rand(k,d)
    max_iters = 1e5
    Y_pred = Y
    np.random.shuffle(Y_pred)   # random initial group membership

    iters = 0
    num_updates = 1
    while (num_updates > 0) and (iters < max_iters):
        plot(X, Y_pred, kMeans, iters)
        # Update centroids based on current membership
        for grp_id in np.unique(Y_pred):
            try:
                kMeans[grp_id] = np.mean(X[np.where(Y_pred==grp_id)[0],:], axis=0)
                
            except:
                import ipdb;ipdb.set_trace()
        
        # Update membership based on current centroids
        num_updates = 0
        for i in range(X.shape[0]):
            x = X[i]

            x_tiled = np.tile(x,(k,1))
            dist_to_kMeans = np.linalg.norm(x_tiled - kMeans, axis=1)
            
            y_pred_old = Y_pred[i]
            y_pred_new = np.argmin(dist_to_kMeans)

            if y_pred_old != y_pred_new:
                Y_pred[i] = y_pred_new
                num_updates += 1
            
        if iters%1 == 0:
            print(f"Iter = {iters} | Num updates: {num_updates}/{X.shape[0]}")

        iters += 1


