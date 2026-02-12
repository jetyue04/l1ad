from typing import Optional, Tuple

import torch
from torch import nn
    

class CouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP.
    
    :param input_dim: Dimension of input features
    :param hidden_dim: Hidden dimension for coupling networks
    :param masking: ParticleMasking module that masks features
    :param conditional_dim: Dimension of conditional context (0 for unconditional)
    :param use_batch_norm: Whether to apply batch normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        masking: nn.Module = None,
        conditional_dim: int = 0,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.conditional_dim = conditional_dim
        self.masking = masking or nn.Identity()
        
        # Scale and translation networks
        network_input_dim = input_dim + conditional_dim
        self.scale_net = nn.Sequential(
            nn.Linear(network_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(network_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.batch_norm = nn.Identity()
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=input_dim)
        
        # Initialize near identity
        if isinstance(self.scale_net[-2], nn.Linear):
            self.scale_net[-2].weight.data.fill_(0.)
            if self.scale_net[-2].bias is not None:
                self.scale_net[-2].bias.data.fill_(0.)
        if isinstance(self.translate_net[-1], nn.Linear):
            self.translate_net[-1].weight.data.fill_(0.)
            if self.translate_net[-1].bias is not None:
                self.translate_net[-1].bias.data.fill_(0.)

    def _compute_scale_translation(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Apply masking and batch norm
        x_masked = self.masking(self.batch_norm(x))
        net_input = x_masked
        if context is not None:
            net_input = torch.cat([x_masked, context], dim=1)            
            
        # Compute scale and translation
        scale = self.scale_net(net_input)
        translation = self.translate_net(net_input)
        
        # Apply masking inverse
        identity_mask = (self.masking(torch.ones_like(x)) == 0).float()
        return scale * identity_mask, translation * identity_mask
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self._compute_scale_translation(x, context)
        s = torch.clamp(s, -2., 2.)
        y = x * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return y, log_det
    
    def inverse(self, y: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self._compute_scale_translation(y, context)
        s = torch.clamp(s, -2., 2.)
        x = (y - t) * torch.exp(-s)
        log_det = -s.sum(dim=1)
        return x, log_det
