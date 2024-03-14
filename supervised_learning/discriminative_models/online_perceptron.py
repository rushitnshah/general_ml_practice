"""
File: online_perceptron.py
Author: Rushit Shah
Date: 03/10/2024
Description: This script implements the Online Perceptron algorithm and 
visualizes the Iris dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from rich import print
from tqdm import tqdm

class OnlineDataGenerator:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.t = 0
    
    def sample(self):
        self.t += 1
        i = np.random.randint(self.N)
        return self.X[i], self.Y[i]

def plot(X, Y, w, xmin, xmax, ymin, ymax):
    w += np.random.normal(0,1e-8,(w.shape[0], ))
    t = X.shape[0]
    x_last = X[-1,:]
    
    _, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=Y)
    scatter = ax.scatter(x_last[0], x_last[1], c='red', marker='x',facecolors='none')
    w_norm = w/np.linalg.norm(w)
    ax.quiver(*[0,0], w_norm[0], w_norm[1], color='r', scale=21)

    x_decision_bdry = np.linspace(xmin, xmax, 10000)
    m = -1*w_norm[0]/w_norm[1]
    y_decision_bdry = m*x_decision_bdry
    ax.plot(x_decision_bdry, y_decision_bdry, 'r:')

    plt.legend(["Data", "Last x"])

    ax.set(xlabel='x1', ylabel='x2')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])


    ax.set_title(f"Timestep {t}")
    plt.savefig(f'./figs/iris_perceptron_step_{t:03d}')
    plt.close()


# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:,[0,2]]
Y = iris.target

# normalize data
mu_X, std_X = X.mean(axis=0), X.std(axis=0)
X = (X - mu_X)/std_X
xmin, xmax = X[:,0].min(), X[:,0].max()
ymin, ymax = X[:,1].min(), X[:,1].max()
xmin -= 0.05*xmin
ymin -= 0.05*ymin
xmax += 0.05*xmax
ymax += 0.05*ymax

inds = np.where(Y != 2)
X, Y = X[inds], Y[inds]

# Initialize the data generator
data_generator = OnlineDataGenerator(X, Y)

# Online Perceptron algorithm
d = X.shape[1]
N = X.shape[0]
T = N
eta = 1/np.sqrt(T)
w0 = np.zeros(d)
margin = 0.5

X_prev = []
Y_prev = []
w = w0
mistakes = 0

# for t in tqdm(range(1, T+1)):
for t in range(1, T+1):
    # Get a single online data sample
    x, y = data_generator.sample()
    X_prev.append(x)
    Y_prev.append(y)
    
    # Relabel negative samples as -1 so that # y\in{0,1} -> y\in{-1,1}
    if y == 0:
        y = -1
    
    # Make prediction
    y_pred = w.dot(x)

    # If prediction incorrect, update w
    if y*y_pred < margin/2:
        w += eta*y*x
        mistakes += 1
    
    plot(np.array(X_prev), np.array(Y_prev), w, xmin, xmax, ymin, ymax)

    if t % 10 == 0:
        print(f"t = {t} | Avg mistakes: {mistakes/t}")

