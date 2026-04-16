from typing import Optional, List
import torch
from torch import nn


class FastFeatureBlur(nn.Module):
    """
    Selectively blurs features by mixing with random values.

    :param p: Probability of applying the noise transformation (0 to 1).
    :param magnitude: Intensity of the blurring effect.
    :param strength: Probability of each feature being affected.
    """

    def __init__(self, prob: float, magnitude: float, strength: float):
        super().__init__()
        self.prob = prob
        self.magnitude = magnitude
        self.strength = strength

    def forward(self, x: torch.Tensor):
        batch_size, feature_dim = x.shape

        # Mask for items to be blurred and features within each item
        mask_p = (torch.rand(batch_size, 1) < self.prob).to(
            device=x.device, dtype=torch.float32
        )
        mask_strength = (torch.rand(batch_size, feature_dim) < self.strength).to(
            device=x.device, dtype=torch.float32
        )

        # Combined mask with magnitude
        mask = mask_p * mask_strength * self.magnitude

        # Apply blurring: original*(1-mask) + random*mask
        return x * (1 - mask) + torch.rand_like(x) * mask


class FastObjectMask(nn.Module):
    """
    Randomly zeros out a subset of features in the input tensor during training.

    :param p: Probability of applying the masking transformation (0 to 1).
    """

    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor):
        # TODO: Check fast object mask implementation
        # This is not clear
        # batch = batch.reshape((-1, 19, 3))
        # mask = torch.rand((batch.shape[0], batch.shape[1]), device=self.device)
        # idx = mask < self.p
        # mask[idx] = 0
        # mask[~idx] = 1
        # batch = batch * mask[:, :, None]
        # batch = batch.reshape((-1, 57))

        if torch.rand(1).item() > self.prob:
            return x

        batch_size, feature_dim = x.shape
        mask = (torch.rand(batch_size, feature_dim) > self.prob).to(
            device=x.device, dtype=torch.float32
        )
        return x * mask


class FastLorentzRotation(nn.Module):
    """
    Applies a random rotation to the phi angles of input features with given probability.

    :param norm_scale: Normalization scale factors
    :param norm_bias: Normalization bias values
    :param p: Probability of applying the rotation to each batch item
    :param phi_indices: Indices of phi angles in the feature vector
    """

    def __init__(
        self,
        prob: float,
        norm_scale: List[float],
        norm_bias: List[float],
        phi_indices: Optional[list] = None,
    ):
        super().__init__()
        self.prob = prob

        # Use register_buffer to store them in the state dictionary but not as model parameters
        phi_indices = phi_indices if phi_indices else torch.arange(0, 19, 1) + 2
        self.register_buffer(
            "l1_scale",
            torch.tensor([144] * 1 + [144] * 4 + [576] * 4 + [144] * 10)
            / (2 * torch.pi),
        )
        self.register_buffer("phi_indices", phi_indices)

        # TODO: Fix scale and bias:
        # self.register_buffer("scale", torch.ones(self.phi_indices.shape[0]).float())
        # self.register_buffer("bias", torch.tensor(self.phi_indices.shape[0]).float())
        self.register_buffer("scale", torch.tensor(norm_scale[:, -1], dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(norm_bias[:, -1], dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        device = x.device
        batch_size = x.shape[0]

        # Create mask for batch items to rotate
        bool_mask = (torch.rand(batch_size, device=device) < self.prob).to(
            device=x.device, dtype=torch.float32
        )

        # Extract and normalize phi values
        original_phi = (x[:, self.phi_indices] * self.scale + self.bias) / self.l1_scale

        # Generate random rotation angles
        rotation = (torch.rand(batch_size, device=device) * 2 * torch.pi)[:, None]

        # Apply rotation and normalize back
        rotated_phi = (
            torch.remainder((original_phi + rotation), 2 * torch.pi)
        ) * self.l1_scale

        # Create new tensor to avoid modifying the input directly
        result = x.clone()

        # Replace values based on the mask
        result[:, self.phi_indices] = (
            (
                bool_mask[:, None] * rotated_phi
                + (1 - bool_mask[:, None]) * original_phi
            ).float()
            - self.bias
        ) / self.scale
        return result


# from typing import Optional, Tuple

# import torch
# from torch import nn
    

# class CouplingLayer(nn.Module):
#     """Affine coupling layer for RealNVP.
    
#     :param input_dim: Dimension of input features
#     :param hidden_dim: Hidden dimension for coupling networks
#     :param masking: ParticleMasking module that masks features
#     :param conditional_dim: Dimension of conditional context (0 for unconditional)
#     :param use_batch_norm: Whether to apply batch normalization
#     """
    
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int = 32,
#         masking: nn.Module = None,
#         conditional_dim: int = 0,
#         use_batch_norm: bool = True
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.conditional_dim = conditional_dim
#         self.masking = masking or nn.Identity()
        
#         # Scale and translation networks
#         network_input_dim = input_dim + conditional_dim
#         self.scale_net = nn.Sequential(
#             nn.Linear(network_input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim),
#             nn.Tanh()
#         )
        
#         self.translate_net = nn.Sequential(
#             nn.Linear(network_input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )
        
#         self.batch_norm = nn.Identity()
#         if use_batch_norm:
#             self.batch_norm = nn.BatchNorm1d(num_features=input_dim)
        
#         # Initialize near identity
#         if isinstance(self.scale_net[-2], nn.Linear):
#             self.scale_net[-2].weight.data.fill_(0.)
#             if self.scale_net[-2].bias is not None:
#                 self.scale_net[-2].bias.data.fill_(0.)
#         if isinstance(self.translate_net[-1], nn.Linear):
#             self.translate_net[-1].weight.data.fill_(0.)
#             if self.translate_net[-1].bias is not None:
#                 self.translate_net[-1].bias.data.fill_(0.)

#     def _compute_scale_translation(
#             self,
#             x: torch.Tensor,
#             context: Optional[torch.Tensor] = None
#         ) -> Tuple[torch.Tensor, torch.Tensor]:

#         # Apply masking and batch norm
#         x_masked = self.masking(self.batch_norm(x))
#         net_input = x_masked
#         if context is not None:
#             net_input = torch.cat([x_masked, context], dim=1)            
            
#         # Compute scale and translation
#         scale = self.scale_net(net_input)
#         translation = self.translate_net(net_input)
        
#         # Apply masking inverse
#         identity_mask = (self.masking(torch.ones_like(x)) == 0).float()
#         return scale * identity_mask, translation * identity_mask
    
#     def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         s, t = self._compute_scale_translation(x, context)
#         s = torch.clamp(s, -2., 2.)
#         y = x * torch.exp(s) + t
#         log_det = s.sum(dim=1)
#         return y, log_det
    
#     def inverse(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         s, t = self._compute_scale_translation(y, context)
#         s = torch.clamp(s, -2., 2.)
#         x = (y - t) * torch.exp(-s)
#         log_det = -s.sum(dim=1)
#         return x, log_det
