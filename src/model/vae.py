import torch.nn as nn
import torch.nn.functional as F
import torch


class VAE_Encoder(nn.Module):
    def __init__(self, input_size, intermediate_architecture, bottleneck_size, drop_out=None):
        super().__init__()

        self.layers = nn.ModuleList()
        architecture = [input_size] + list(intermediate_architecture)
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
        self.n_layers = len(self.layers)

        self.layer_mu      = nn.Linear(intermediate_architecture[-1], bottleneck_size)
        self.layer_log_var = nn.Linear(intermediate_architecture[-1], bottleneck_size)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = F.relu(layer(x))
        return self.layer_mu(x), self.layer_log_var(x)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    @staticmethod
    def loss(x, x_hat, mu, log_var):
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from .encoder import Encoder

# class VAE_Encoder(nn.Module):
#     def __init__(self, input_size, intermediate_architecture, bottleneck_size, drop_out=None):
#         super().__init__()

#         self.trunk = Encoder(
#             input_size=input_size,
#             intermediate_architecture=intermediate_architecture[:-1],
#             bottleneck_size=intermediate_architecture[-1],
#             drop_out=drop_out
#         )
#         self.layer_mu      = nn.Linear(intermediate_architecture[-1], bottleneck_size)
#         self.layer_log_var = nn.Linear(intermediate_architecture[-1], bottleneck_size)

#     def forward(self, x):
#         z = self.trunk(x)
#         return self.layer_mu(z), self.layer_log_var(z)


# class VariationalAutoEncoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + std * eps

#     def forward(self, x):
#         mu, log_var = self.encoder(x)
#         z = self.reparameterize(mu, log_var)
#         x_hat = self.decoder(z)
#         return x_hat, mu, log_var

#     @staticmethod
#     def loss(x, x_hat, mu, log_var):
#         reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
#         kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
#         total_loss = reconstruction_loss + kl_loss
#         return total_loss, reconstruction_loss, kl_loss
#     # @staticmethod
#     # def loss(x, x_hat, mu, log_var, kl_weight=1.0):
#     #     reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
#     #     kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
#     #     total_loss = reconstruction_loss + kl_weight * kl_loss
#     #     return total_loss, reconstruction_loss, kl_loss