import torch
from torch import nn


class FastFeatureBlur(nn.Module):
    def __init__(self, prob: float, magnitude: float, strength: float):
        super().__init__()
        self.prob = prob
        self.magnitude = magnitude
        self.strength = strength

    def forward(self, x: torch.Tensor):
        batch = x.reshape(-1, 19, 3)

        mask_p = torch.rand(batch.shape[0], 1, 1, device=x.device)
        mask_p = (mask_p < self.prob).float()

        mask_strength = (torch.rand_like(batch) < self.strength).float()

        mask_magnitude = torch.ones_like(batch) * self.magnitude * mask_strength * mask_p

        blur = torch.rand_like(batch)
        batch = batch * (1 - mask_magnitude) + blur * mask_magnitude

        return batch.reshape(-1, 57)


class FastObjectMask(nn.Module):
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor):
        batch = x.reshape(-1, 19, 3)
        mask = torch.rand(batch.shape[0], batch.shape[1], device=x.device)
        mask = (mask >= self.prob).float()
        batch = batch * mask[:, :, None]
        return batch.reshape(-1, 57)


class FastLorentzRotation(nn.Module):
    def __init__(self, prob, norm_scale, norm_bias, phi_indices=None):
        super().__init__()
        self.prob = prob
        phi_indices = torch.arange(0, 19, 1) + 2  # [2,3,4,...,20]
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

        bool_mask = torch.rand(batch_size, device=device)
        idx = bool_mask < self.prob
        bool_mask[idx] = 1
        bool_mask[~idx] = 0

        original_phi = (x[:, self.phi_indices] * self.scale + self.bias) / self.l1_scale
        rotation = (torch.rand(batch_size, device=device) * 2 * torch.pi)[:, None]
        rotated_phi = torch.remainder(original_phi + rotation, 2 * torch.pi) * self.l1_scale

        result = x.clone()
        result[:, self.phi_indices] = (
            (bool_mask[:, None] * rotated_phi + (1 - bool_mask[:, None]) * original_phi).float()
            - self.bias
        ) / self.scale
        return result
