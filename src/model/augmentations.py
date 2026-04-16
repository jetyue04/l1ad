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
    Zeros out entire objects (all 3 features of a particle) with probability `prob`.
    Fixed: original used the same prob for both an outer gate and per-feature masking,
    and masked individual features rather than whole objects.
    """
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor):
        batch_size, feature_dim = x.shape
        n_objects = feature_dim // 3
        x_3d = x.view(batch_size, n_objects, 3)
        obj_mask = (torch.rand(batch_size, n_objects, device=x.device) >= self.prob).float()
        return (x_3d * obj_mask.unsqueeze(-1)).view(batch_size, feature_dim)

class FastLorentzRotation(nn.Module):
    """
    Applies random phi rotation to each particle's phi feature.
    Fixed two bugs from src/model/augmentations.py:
      1. phi_indices were consecutive [2..20] instead of stride-3 [2,5,8,...,56].
      2. non-rotation branch used original_phi (radians) in place of the already-normalized
         x values, corrupting untouched events.
    """
    def __init__(self, prob, norm_scale, norm_bias, phi_indices=None):
        super().__init__()
        self.prob = prob
        phi_indices = phi_indices if phi_indices is not None else torch.arange(0, 19) * 3 + 2
        self.register_buffer(
            "l1_scale",
            torch.tensor([144] * 1 + [144] * 4 + [576] * 4 + [144] * 10) / (2 * torch.pi),
        )
        self.register_buffer("phi_indices", phi_indices)
        self.register_buffer("scale", torch.tensor(norm_scale[:, -1], dtype=torch.float32))
        self.register_buffer("bias",  torch.tensor(norm_bias[:, -1],  dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        device = x.device
        batch_size = x.shape[0]
        bool_mask = (torch.rand(batch_size, device=device) < self.prob).float()

        # normalized -> raw phi -> radians
        original_phi = (x[:, self.phi_indices] * self.scale + self.bias) / self.l1_scale

        # rotate in radian space, then renormalize: radians -> raw phi -> normalized
        rotation = (torch.rand(batch_size, device=device) * 2 * torch.pi)[:, None]
        rotated_normalized = (
            torch.remainder(original_phi + rotation, 2 * torch.pi) * self.l1_scale
            - self.bias
        ) / self.scale

        result = x.clone()
        result[:, self.phi_indices] = (
            bool_mask[:, None] * rotated_normalized
            + (1 - bool_mask[:, None]) * x[:, self.phi_indices]
        )
        return result