import argparse

import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import generate_synthetic_data, CustomDatasetWithShape, plot
# from models.vae import VAE
# from gan.gan import GAN
from models import GAN, VAE
from models.gan2 import GAN2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='vae')
    parser.add_argument('--nepochs', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--plot_at_eval', action='store_true')
    args = parser.parse_args()

    # Hyperparams
    nsamples = 1000
    nbatches = 10
    batch_size = nsamples // nbatches
    train_frac = 0.8
    nepochs = args.nepochs
    eval_freq = args.eval_freq
    
    # Generate synthetic data
    x_train, x_test = generate_synthetic_data(nsamples, train_frac)
    full_dataset = th.concatenate((x_train, x_test),dim=0)

    # Create Dataloaders
    dataset_train = CustomDatasetWithShape(x_train)
    dataset_test = CustomDatasetWithShape(x_test)
    ndims = dataset_train.shape[1]
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    if args.algo == 'vae':
        # Build model
        ndims_hid=[8] # [16,8]
        model = VAE(ndims_in=ndims, ndims_hid=[8], ndims_latent=2, lr=0.001, device=None)
    elif args.algo == 'gan':
        # Data Dims and Network Hidden Dims
        ndims_data = dataset_train.shape[-1]
        ndims_hid = [32,16]

        # Generator Dims
        # ndims_hid=[16,32,16]
        ndims_hid=[256,128,64]
        gen_ndims_in = 2
        gen_ndims_out = ndims_data
        gen_dims = [gen_ndims_in, *ndims_hid, gen_ndims_out]

        # Discriminator Dims
        ndims_hid=[256,128,64]
        disc_ndims_in = ndims_data
        disc_ndims_out = 1
        disc_dims = [disc_ndims_in, *ndims_hid, disc_ndims_out]

        model = GAN(gen_dims, disc_dims, lr=0.001, device=None)
    
    # Train
    model, train_loss_hist, test_loss_hist, x_pred_bef, x_pred_aft = model.learn(
            train_dataloader, test_dataloader, full_dataset, 
            nepochs=nepochs, eval_freq=eval_freq, plot_at_eval=args.plot_at_eval
    )

    # Save final Plot
    plot(
        full_dataset, x_pred_bef, x_pred_aft,
        train_loss_hist, test_loss_hist, 
        epoch=nepochs, savefig=True
    )

if __name__=='__main__':
    main()