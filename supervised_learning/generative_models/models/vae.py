import torch as th
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from utils import plot

class VAE(BaseModel):
    def __init__(self, ndims_in, ndims_hid=[16,8], ndims_latent=2, lr=0.001,
            device=None):
        super(VAE, self).__init__()
        self.name = 'vae'
        self.ndims_in = ndims_in
        self.ndims_out = ndims_in
        self.ndims_hid = ndims_hid
        self.ndims_latent = ndims_latent
        self.lr = lr
        self.reparam_noise = 1e-6

        self._build()

        self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _build(self):
        # Build encoder layer dims
        curr_dim = self.ndims_hid[0]
        enc_layers = [nn.Linear(self.ndims_in, curr_dim), nn.ReLU()]
        if len(self.ndims_hid) > 1:
            for prev_dim, curr_dim in zip(self.ndims_hid[:-1], self.ndims_hid[1:]):
                enc_layers.extend(
                    [nn.Linear(prev_dim, curr_dim),nn.ReLU()]
                )
        self.encoder = th.nn.Sequential(*enc_layers)
        # prev_dim = enc_layers[-1].out_features
        prev_dim = curr_dim

        # Build latent layer dims
        latent_dim_in = prev_dim

        # Build decoder layer dims
        curr_dim = self.ndims_hid[-1]
        dec_layers = [nn.Linear(self.ndims_latent, curr_dim), nn.ReLU()]
        if len(self.ndims_hid) > 1:
            for prev_dim, curr_dim in zip(self.ndims_hid[1:][::-1], self.ndims_hid[:-1][::-1]):
                dec_layers.extend(
                    [nn.Linear(prev_dim, curr_dim),nn.ReLU()]
                )
        # prev_dim = enc_layers[-1].out_features
        prev_dim = curr_dim
        dec_layers.append(nn.Linear(prev_dim, self.ndims_out))

        # initiazlie layers
        self.encoder = th.nn.Sequential(*enc_layers)
        self.latent_mu = nn.Linear(latent_dim_in, self.ndims_latent)
        self.latent_sigma = nn.Linear(latent_dim_in, self.ndims_latent)
        self.decoder = th.nn.Sequential(*dec_layers)

    
    def forward(self, x, reparameterize=True):
        # Encode
        x = self._to_tensor(x)
        x = self.encoder(x)

        # Sample latent representation
        mu = self.latent_mu(x)
        sigma = self.latent_sigma(x)
        sigma = th.clamp(sigma, min=self.reparam_noise)
        dist = th.distributions.Normal(mu, sigma)
        if reparameterize:
            z = dist.rsample()
        else:
            z = dist.sample()
        
        # Decode
        x = self.decoder(z)
        return x

    def generate(self, x_true, reparameterize):
        out = self.forward(x_true, reparameterize=reparameterize)
        return out
    
    @th.no_grad()
    def estimate_loss(self, dataloader):
        self.eval()
        num_eval_batches = int(len(dataloader.dataset)/dataloader.batch_size)
        losses = th.zeros(num_eval_batches)
        for k, x_true in enumerate(dataloader):
            x_pred = self(x_true, reparameterize=True)
            loss = F.mse_loss(x_pred, x_true)
            losses[k] = loss.item()
        loss = losses.mean()
        self.train()
        return loss

    def learn(self, train_dataloader, test_dataloader, full_dataset, 
        nepochs: int = 10, eval_freq: int = 1, plot_at_eval=False):
        train_loss_hist = []
        test_loss_hist = []

        # Compute predictions BEFORE training
        with th.no_grad():
            x_pred_bef = self.generate(full_dataset, reparameterize=False)
        
        #  ~~~~~ TRAIN LOOP ~~~~~  #
        for i in range(nepochs):
            # Eval
            if i % eval_freq == 0:
                train_loss = self.estimate_loss(train_dataloader)
                train_loss_hist.append(train_loss)
                test_loss = self.estimate_loss(test_dataloader)
                test_loss_hist.append(test_loss)
                if plot_at_eval:
                    with th.no_grad():
                        x_pred_aft = self.generate(full_dataset, reparameterize=False)
                    plot(full_dataset, x_pred_bef, x_pred_aft, train_loss_hist,
                        test_loss_hist, epoch=i, savefig=True, modeldir=self.name)
                self.print_to_terminal(epoch=i, train_loss=train_loss, test_loss=test_loss)
            
            # Train
            for x_true in train_dataloader:
                x_pred = self(x_true, reparameterize=True)
                loss = F.mse_loss(x_pred, x_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self, train_loss_hist, test_loss_hist, x_pred_bef, x_pred_aft

        