import torch
import torch.nn as nn

class VICRegProjector(nn.Module):
    def __init__(self, input_dim, projection_dim=128, num_layers=3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, projection_dim))
            layers.append(nn.BatchNorm1d(projection_dim))
            layers.append(nn.ReLU())
            prev_dim = projection_dim

        layers.append(nn.Linear(prev_dim, projection_dim))
        layers.append(nn.BatchNorm1d(projection_dim))

        self.model = nn.Sequential(*layers)
        self.output_dim = projection_dim

    def forward(self, x):
        return self.model(x)

class VICReg(nn.Module):
    def __init__(
        self,
        encoder,
        projection_dim=128,
        projection_layers=3,
        sim_coeff=50,
        std_coeff=50,
        cov_coeff=1,
    ):
        super().__init__()

        self.encoder = encoder
        self.projector = VICRegProjector(
            input_dim=encoder.layers[-1].out_features,
            projection_dim=projection_dim,
            num_layers=projection_layers
        )

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z
