import torch as th
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from utils import plot

# class GeneratorCopy(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(2, 16),
#             nn.ReLU(),
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2),
#         )

#     def forward(self, x):
#         output = self.model(x)
#         return output

# class DiscriminatorCopy(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(2, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         output = self.model(x)
#         return output

class Generator(th.nn.Module):
    '''
    Reference explanation for implementation
    https://realpython.com/generative-adversarial-networks/
    '''
    def __init__(self, dims: list):
        super().__init__()
        layers = []
        for prev_dim, curr_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                [nn.Linear(prev_dim, curr_dim), nn.ReLU(), nn.Dropout(0.3)]
            )
        layers.append(nn.Linear(curr_dim, dims[-1]))
        self.model = th.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(th.nn.Module):
    def __init__(self, dims: list):
        super().__init__()
        layers = []
        for prev_dim, curr_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                [nn.Linear(prev_dim, curr_dim), nn.ReLU()]
            )
        layers.append(nn.Linear(curr_dim, dims[-1]))
        layers.append(nn.Sigmoid())
        self.model = th.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class GAN(BaseModel):
    def __init__(self, gen_dims, disc_dims, lr=0.001, device=None):
        super(GAN, self).__init__()
        self.name = 'gan'
        self.gen_dims = gen_dims
        self.disc_dims = disc_dims
        self.lr = lr
        self.disc_dims = disc_dims
        self.gen_dim_out = self.gen_dims[-1]

        self._build()

        self.optimizer_gen = th.optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizer_disc = th.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _build(self):
        # self.generator = Generator()
        # self.discriminator = Discriminator()
        self.generator = self._build_network_from_dims(self.gen_dims)
        self.discriminator = self._build_network_from_dims(self.disc_dims, output_probs=True)
        # self.generator = Generator(self.gen_dims)
        # self.discriminator = Discriminator(self.disc_dims)
    
    def _build_network_from_dims(self, dims, output_probs=False):
        layers = []
        for prev_dim, curr_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                [nn.Linear(prev_dim, curr_dim), nn.ReLU()]
            )
        layers.append(nn.Linear(curr_dim, dims[-1]))
        if output_probs:
            layers.append(nn.Sigmoid())
        return th.nn.Sequential(*layers)
    
    def generate(self, batch_size, z=None):
        if z is None:
            z = th.rand(batch_size, self.gen_dim_out)
        return self.generator(z), z

    def discriminate(self, x):
        x = self._to_tensor(x)
        return self.discriminator(x)

    def learn(self, train_dataloader, test_dataloader, full_dataset, 
        nepochs: int = 10, eval_freq: int = 1, plot_at_eval=False):
        train_loss_hist_gen = []
        train_loss_hist_disc = []
        test_loss_hist_gen = []
        test_loss_hist_disc = []
        n_samples_total = full_dataset.shape[0]
        
        batch_size_train = train_dataloader.batch_size
        
        # Compute predictions BEFORE training
        with th.no_grad():
            x_pred_bef, _ = self.generate(batch_size=n_samples_total)
        self.loss_criterion = nn.BCELoss()
        #  ~~~~~ TRAIN LOOP ~~~~~  #
        for i in range(nepochs):
            # Eval
            if i % eval_freq == 0:
                train_loss_gen, train_loss_disc = self.estimate_loss(train_dataloader)
                train_loss_hist_gen.append(train_loss_gen)
                train_loss_hist_disc.append(train_loss_disc)
                
                test_loss_gen, test_loss_disc  = self.estimate_loss(test_dataloader)
                test_loss_hist_gen.append(test_loss_gen)
                test_loss_hist_disc.append(test_loss_disc)
                if plot_at_eval:
                    with th.no_grad():
                        x_pred_aft, _  = self.generate(batch_size=n_samples_total)
                    plot(full_dataset, x_pred_bef, x_pred_aft, train_loss_hist_gen,
                        test_loss_hist_gen, epoch=i, savefig=True, modeldir=self.name)
                self.print_to_terminal(i, train_loss_gen, train_loss_disc,
                                                test_loss_gen, test_loss_disc)
            
            # Train
            for x_true in train_dataloader:
                z = th.randn((batch_size_train, 2))
                x_pred, _ = self.generate(batch_size=batch_size_train, z=z)
                # x_pred, _ = self.generate(batch_size=batch_size_train)
                # x_pred = self.generator(z)
                y_true = th.ones(batch_size_train)
                y_pred = th.zeros(batch_size_train)
                
                x = th.cat((x_true,x_pred)).view(2*batch_size_train,-1)
                y = th.cat((y_true,y_pred)).view(2*batch_size_train,-1)
                
                # Discriminator Update
                y_disc = self.discriminate(x)
                # y_disc = self.discriminator(x)
                loss_disc = F.binary_cross_entropy(y_disc, y)
                # loss_disc = self.loss_criterion(y_disc, y)
                self.optimizer_disc.zero_grad()
                loss_disc.backward()
                self.optimizer_disc.step()

                # Generator Update
                x_pred, _ = self.generate(batch_size=batch_size_train, z=z)
                # x_pred, _ = self.generate(batch_size=batch_size_train)
                # x_pred = self.generator(z)
                y_disc_pred = self.discriminate(x_pred)
                # y_disc_pred = self.discriminator(x_pred)
                # loss_gen = F.binary_cross_entropy(y_disc_pred, y_pred.view(batch_size_train,-1))
                loss_gen = F.binary_cross_entropy(y_disc_pred, y_true.view(batch_size_train,-1))
                
                self.optimizer_gen.zero_grad()
                loss_gen.backward()
                self.optimizer_gen.step()
            
            # if i % 10 == 0:
            #     print(f"epoch {i}: loss_gen: {loss_gen.item()}, loss_disc: {loss_disc.item()}")
        return self, train_loss_hist_gen, test_loss_hist_gen, x_pred_bef, x_pred_aft

    @th.no_grad()
    def estimate_loss(self, dataloader):
        self.eval()
        batch_size = dataloader.batch_size
        num_eval_batches = int(len(dataloader.dataset)/batch_size)
        losses_gen = th.zeros(num_eval_batches)
        losses_disc = th.zeros(num_eval_batches)
        for k, x_true in enumerate(dataloader):
            x_pred, _ = self.generate(batch_size=batch_size)
            y_true = th.ones(batch_size)
            y_pred = th.zeros(batch_size)
            
            x = th.cat((x_true,x_pred)).view(2*batch_size,-1)
            y = th.cat((y_true,y_pred)).view(2*batch_size,-1)

            # Discriminator Loss
            y_disc = self.discriminate(x)
            loss_disc = F.binary_cross_entropy(y_disc, y)
            losses_disc[k] = loss_disc.item()

            # Generator Update
            x_pred, _ = self.generate(batch_size=batch_size)
            y_disc_pred = self.discriminate(x_pred)
            loss_gen = F.binary_cross_entropy(y_disc_pred, y_pred.view(batch_size,-1))
            losses_gen[k] = loss_gen.item()
        loss_gen = losses_gen.mean()
        loss_disc = losses_disc.mean()
        self.train()
        return loss_gen, loss_disc
    
    def print_to_terminal(self, epoch, train_loss_gen, train_loss_disc, 
                test_loss_gen, test_loss_disc):
        if epoch == 0:
            print("|  epoch   |  train loss (gen/disc)  |  test loss (gen/disc)  |")
        prnt_str = f"|    {epoch:5d}  "
        prnt_str += f"|  {train_loss_gen:.4f} / {train_loss_disc:.4f}   "
        prnt_str += f"|  {test_loss_gen:.4f} / {test_loss_disc:.4f}   |"
        print(prnt_str)

    
    # def new_learn(self, train_dataloader, test_dataloader, full_dataset, 
    #     nepochs: int = 10, eval_freq: int = 1, plot_at_eval=False):
    #     batch_size_train = train_dataloader.batch_size
    #     loss_criterion = nn.BCELoss()
    #     for i in range(nepochs):
    #         for n, real_samples in enumerate(train_dataloader):
    #             real_samples_labels = th.ones((batch_size_train,1))
    #             generated_samples_labels = th.zeros((batch_size_train,1))
    #             all_samples_labels = th.cat((real_samples_labels, generated_samples_labels))
                
    #             # generated_samples, latent_space_samples = self.generate(batch_size_train)
    #             latent_space_samples = th.randn((batch_size_train, 2))
    #             generated_samples = self.generator(latent_space_samples)
    #             all_samples = th.cat((real_samples, generated_samples))

    #             # Train discriminator
    #             self.optimizer_disc.zero_grad()
    #             # output_discriminator = self.discriminate(all_samples)
    #             output_discriminator = self.discriminator(all_samples)
    #             loss_discriminator = loss_criterion(output_discriminator, all_samples_labels)
    #             loss_discriminator.backward()
    #             self.optimizer_disc.step()

    #             # Train generator
    #             self.optimizer_gen.zero_grad()
    #             # generated_samples, latent_space_samples = self.generate(batch_size_train)
    #             generated_samples = self.generator(latent_space_samples)
    #             # output_discriminator = self.discriminate(generated_samples)
    #             output_discriminator = self.discriminator(generated_samples)
    #             loss_generator = loss_criterion(output_discriminator, real_samples_labels)
    #             loss_generator.backward()
    #             self.optimizer_gen.step()
            
    #         if i % eval_freq == 0:
    #             print(f"Epoch: {i:5d} | L_D: {loss_discriminator.item():.4f}   |   L_G: {loss_generator.item():.4f}   |")



    # def learn_combined(self, train_dataloader, test_dataloader, full_dataset, 
    #     # Loss = loss_gen + loss_disc
    #     nepochs: int = 10, eval_freq: int = 1, plot_at_eval=False):
    #     train_loss_hist_gen = []
    #     train_loss_hist_disc = []
    #     test_loss_hist_gen = []
    #     test_loss_hist_disc = []
    #     n_samples_total = full_dataset.shape[0]
        
    #     batch_size_train = train_dataloader.batch_size
        
    #     # Compute predictions BEFORE training
    #     with th.no_grad():
    #         x_pred_bef, _ = self.generate(batch_size=n_samples_total)
    #     self.loss_criterion = nn.BCELoss()
    #     #  ~~~~~ TRAIN LOOP ~~~~~  #
    #     for i in range(nepochs):
    #         # Train
    #         for x_true in train_dataloader:
    #             z = th.randn((batch_size_train, 2))
    #             x_pred, _ = self.generate(batch_size=batch_size_train, z=z)
    #             # x_pred, _ = self.generate(batch_size=batch_size_train)
    #             # x_pred = self.generator(z)
    #             y_true = th.ones(batch_size_train)
    #             y_pred = th.zeros(batch_size_train)
                
    #             x = th.cat((x_true,x_pred)).view(2*batch_size_train,-1)
    #             y = th.cat((y_true,y_pred)).view(2*batch_size_train,-1)
                
    #             # Discriminator Loss
    #             y_disc = self.discriminate(x)
    #             # y_disc = self.discriminator(x)
    #             loss_disc = F.binary_cross_entropy(y_disc, y)
    #             # loss_disc = self.loss_criterion(y_disc, y)

    #             # Generator Update
    #             # x_pred, _ = self.generate(batch_size=batch_size_train, z=z)
    #             # x_pred, _ = self.generate(batch_size=batch_size_train)
    #             # x_pred = self.generator(z)
    #             y_disc_pred = self.discriminate(x_pred)
    #             # y_disc_pred = self.discriminator(x_pred)
    #             # loss_gen = F.binary_cross_entropy(y_disc_pred, y_pred.view(batch_size_train,-1))
    #             loss_gen = F.binary_cross_entropy(y_disc_pred, y_true.view(batch_size_train,-1))
                
    #             loss = loss_gen + loss_disc
    #             self.optimizer_disc.zero_grad()
    #             self.optimizer_gen.zero_grad()
    #             loss.backward()
    #             self.optimizer_disc.step()
    #             self.optimizer_gen.step()
            
    #         if i % 10 == 0:
    #             print(f"epoch {i}: loss_gen: {loss_gen.item()}, loss_disc: {loss_disc.item()}")
    #     return self, train_loss_hist_gen, test_loss_hist_gen, x_pred_bef, x_pred_aft