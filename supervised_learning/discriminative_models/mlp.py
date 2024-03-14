import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MlpClassifier(th.nn.Module):
    def __init__(self, ndims_in, ndims_out, ndims_hid=[64,64], lr=0.01, 
        device='auto', dropout=0.0, output_probs=False):
        super().__init__()
        self.ndims_in = ndims_in
        self.ndims_out = ndims_out
        self.ndims_hid = ndims_hid
        self.lr = lr
        if device=='auto':
            self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        else:
            self.device = th.device(device)
        self.dropout = dropout
        self.output_probs = output_probs

        self.build_network()
        self.to(self.device, dtype=th.float)
        self.optim = th.optim.Adam(self.parameters(), lr=self.lr)
    
    def build_network(self):
        layers = []
        last_layer_dim = self.ndims_in
        for next_layer_dim in self.ndims_hid:
            layers.append(nn.Linear(last_layer_dim, next_layer_dim))
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            layers.append(th.nn.ReLU())
            last_layer_dim = next_layer_dim
        
        layers.append(nn.Linear(last_layer_dim, self.ndims_out))
        if self.output_probs:
            layers.append(th.nn.Softmax())
        
        self.network = th.nn.Sequential(*layers)
        # self.fc1 = nn.Linear(self.ndims_in, self.ndims_hid[0])
        # self.fc2 = nn.Linear(self.ndims_hid[0], self.ndims_out)
        # print(self.network)
    
    def preprocess_obs(self, x):
        if isinstance(x, np.ndarray):
            x = th.tensor(x, device=self.device, dtype=th.float)
        else:
            x = x.to(self.device, dtype=th.float)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(axis=0)
        return x
    
    def forward(self, x):
        x = self.preprocess_obs(x)
        out = self.network(x)
        return out
        # out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        # return out
    
    def predict(self, x):
        out =  self.forward(x)
        if not self.output_probs:
            probs = F.softmax(out, dim=1)
        else:
            probs = out
        y_pred = th.argmax(probs, axis=1).cpu().detach().numpy()
        return y_pred

class IrisDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.nsamples = X.shape[0]
    
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train():
    np.random.seed(42)
    th.manual_seed(42)

    # Hyperparams
    max_epochs = 100
    test_size = 0.2
    batch_size = 10
    eval_freq = 10
    dropout = 0.25
    ndims_hid = [128,128]
    
    # Prepare data
    iris = load_iris()
    X, Y = iris.data, iris.target
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=test_size)
    dataloader_tr = DataLoader(IrisDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    dataloader_te = DataLoader(IrisDataset(X_te, Y_te), batch_size=batch_size, shuffle=True)

    nsamples, ndims_in = X.shape
    ndims_out = np.unique(Y).size

    # Initialize model
    model = MlpClassifier(ndims_in=ndims_in, ndims_out=ndims_out, 
                        ndims_hid=ndims_hid, dropout=dropout)

    # Train loop
    ep_loss_hist = np.zeros(max_epochs)
    ep_acc_hist_tr = np.zeros(max_epochs)
    ep_acc_hist_te = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        curr_ep_losses = []
        curr_ep_accs_tr = []
        curr_ep_accs_te = []
        for x, y in dataloader_tr:
            # Predict output
            output = model(x)
            
            # Compute loss
            loss = F.cross_entropy(output, y)

            # Backprob
            model.optim.zero_grad()
            loss.backward()
            model.optim.step()

            # Evaluate/update histories
            curr_ep_losses.append(loss.detach().cpu().item())
            curr_ep_accs_tr.append(evaluate(model, dataloader_tr))
            curr_ep_accs_te.append(evaluate(model, dataloader_te))
        ep_loss_hist[epoch] = np.mean(curr_ep_losses)
        ep_acc_hist_tr[epoch] = np.mean(curr_ep_accs_tr)
        ep_acc_hist_te[epoch] = np.mean(curr_ep_accs_te)

        if epoch % eval_freq == 0:
            print(f"epoch: {epoch:03d}   |  loss: {ep_loss_hist[epoch]:04.4f}   "+\
                f"|  acc_tr: {ep_acc_hist_tr[epoch]:02.2f}   "+\
                f"|  acc_te: {ep_acc_hist_te[epoch]:02.2f}")
            

def evaluate(model, dataloader):
    model.network.eval()
    accs = []
    for x, y in dataloader:
        y_pred = model.predict(x)
        accs.append((y_pred == y.numpy()).astype(np.int8).sum()/y.shape[0])
    model.network.train()
    return np.mean(accs)


if __name__=='__main__':
    train()

