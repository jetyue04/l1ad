import numpy as np
import torch
import ot

from wnae._sample_buffer import SampleBuffer
from wnae._mcmc_utils import sample_langevin
from wnae._logger import log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WNAE(torch.nn.Module):
    """Wasserstein Normalized Autoencoder.

    Args:
        encoder (torch.nn.Module)
        decoder (torch.nn.Module)
        sampling (str): Sampling methods, choose from 'cd' (Contrastive
            Divergence), 'pcd' (Persistent CD), 'omi' (on-manifold
            initialization).
        x_step (int): Number of steps in MCMC.
        x_step_size (float): Step size of MCMC.
        x_noise_std (float): The standard deviation of noise in MCMC.
        x_temperature (float): Temperature of the MCMC.
        x_bound (tuple or None): Tuple of two floats with min and max values
            to clip the MCMC samples. If None, samples will not be clipped.
        x_clip_grad (tuple or None): Tuple of two floats with min and max values
            to clip the gradient norm. If None, gradient will not be clipped.
        x_mh (bool): If True, use Metropolis-Hastings rejection in MCMC.
        z_step (int): Same as `x_step` but for the latent MCMC.
        z_step_size (float): Same as `x_step_size` but for the latent MCMC.
        z_noise_std (float): Same as `x_noise_std` but for the latent MCMC.
        z_temperature (float): Same as `x_temperature` but for the latent MCMC.
        z_bound (tuple or None): Same as `x_bound` but for the latent MCMC.
        z_clip_grad (tuple or None): Same as `x_clip_grad` but for the latent MCMC.
        z_mh (bool): Same as `x_mh` but for the latent MCMC.
        spherical (bool): Project latent vectors onto the hypersphere.
        initial_dist (str): Distribution from which initial samples are
            generated. Choose from 'gaussian' or 'uniform'.
        replay (bool): Whether to use the replay buffer.
        replay_ratio (float): For PCD, probability to keep sample for
            the initialization of the next chain.
        buffer_size (int): Size of replay buffer.
    """

    def __init__(
        self,
        encoder,
        decoder,
        sampling="pcd",
        x_step=50,
        x_step_size=10,
        x_noise_std=0.05,
        x_temperature=1.0,
        x_bound=(0, 1),
        x_clip_grad=None,
        x_reject_boundary=False,
        x_mh=False,
        z_step=50,
        z_step_size=0.2,
        z_noise_std=0.2,
        z_temperature=1.0,
        z_bound=None,
        z_clip_grad=None,
        z_reject_boundary=False,
        z_mh=False,
        spherical=False,
        initial_dist="gaussian",
        replay=True,
        replay_ratio=0.95,
        buffer_size=10000,
    ):
        super().__init__()

        x_step_size, x_noise_std = self.__mcmc_checks_and_definition(x_step_size, x_noise_std)
        z_step_size, z_noise_std = self.__mcmc_checks_and_definition(z_step_size, z_noise_std)

        self.encoder = encoder
        self.decoder = decoder
        self.sampling = sampling
        
        self.x_step = x_step
        self.x_step_size = x_step_size
        self.x_noise_std = x_noise_std
        self.x_temperature = x_temperature
        self.x_bound = x_bound
        self.x_clip_grad = x_clip_grad
        self.x_mh = x_mh
        self.x_reject_boundary = x_reject_boundary

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.replay = replay

        self.buffer = SampleBuffer(max_samples=buffer_size, replay_ratio=replay_ratio)

        self.initial_dist = initial_dist

        assert z_step_size is not None or z_noise_std is not None
        assert z_step_size is None or z_step_size > 0

        if z_step_size is None or z_noise_std is None:
            if z_step_size is None:
                z_step_size = z_noise_std**2 / 2.
            else:
                z_noise_std = np.sqrt(2 * z_step_size)


        self.z_step = z_step
        self.z_step_size = z_step_size
        self.z_noise_std = z_noise_std
        self.z_temperature = z_temperature
        self.z_bound = z_bound
        self.z_clip_grad = z_clip_grad
        self.z_mh = z_mh
        self.z_reject_boundary = z_reject_boundary

        self.spherical = spherical

        self.z_shape = None
        self.x_shape = None


    @property
    def sample_shape(self):
        if self.sampling == "omi":
            return self.z_shape
        else:
            return self.x_shape

    @staticmethod
    def __mcmc_checks_and_definition(step_size, noise_std):
        assert step_size is not None or noise_std is not None
        assert step_size is None or step_size > 0

        if step_size is None or noise_std is None:
            if step_size is None:
                step_size = noise_std**2 / 2.
            else:
                noise_std = np.sqrt(2 * step_size)

        return step_size, noise_std

    @staticmethod
    def __mse(y_true, y_pred):
        """Mean Squared Error (MSE).

        Args:
            y_true (torch.Tensor)
            y_pred (torch.Tensor)
        """

        n_dim = np.prod(y_true.shape[1:])

        return ((y_true - y_pred) ** 2).view((y_true.shape[0], -1)).sum(dim=1) / n_dim
    
    @staticmethod
    def __mae(y_true, y_pred):
        """Mean Squared Error (MSE).

        Args:
            y_true (torch.Tensor)
            y_pred (torch.Tensor)
        """

        n_dim = np.prod(y_true.shape[1:])

        return (abs(y_true - y_pred)).view((y_true.shape[0], -1)).sum(dim=1) / n_dim

    def error(self, x, recon, score_method = "mse"):
        if score_method == "mae":
            return self.__mae(x, recon)
        else:
            return self.__mse(x, recon)

    def __normalize(self, z):
        """normalize to unit length"""
        if self.spherical:
            if len(z.shape) == 4:
                z = z / z.view(len(z), -1).norm(dim=-1)[:, None, None, None]
            else:
                z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)
            return z
        else:
            return z

    def encode(self, x):
        if self.spherical:
            return self.__normalize(self.encoder(x))
        else:
            return self.encoder(x)

    def forward(self, x, score_method="mse"):
        """Computes error"""

        z = self.encode(x)
        recon = self.decoder(z)
        return self.error(x, recon, score_method=score_method)
    
    def energy(self, x):
        return self.forward(x)

    def energy_with_z(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return self.error(x, recon), z

    def __set_x_shape(self, x):
        if self.x_shape is not None:
            return
        self.x_shape = x.shape[1:]

    def __set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def __set_shapes(self, x):
        self.__set_z_shape(x)
        self.__set_x_shape(x)

    def __initial_sample(self, n_samples, device):
        l_sample = []
        if not self.replay or len(self.buffer) == 0:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_samples) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_samples - n_replay,) + self.sample_shape
        if self.initial_dist == "gaussian":
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == "uniform":
            x0_new = torch.rand(shape, dtype=torch.float)
            if self.sampling != "omi" and self.x_bound is not None:
                x0_new = x0_new * (self.x_bound[1] - self.x_bound[0]) + self.x_bound[0]
            elif self.sampling == "omi" and self.z_bound is not None:
                x0_new = x0_new * (self.z_bound[1] - self.z_bound[0]) + self.z_bound[0]
        else:
            log.critical(f"Invalid initial distribution {self.initial_dist}")
            exit(1)

        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)

    def __sample_x(self, n_sample=None, device=None, x0=None, replay=False):
        if x0 is None:
            x0 = self.__initial_sample(n_sample, device=device)
        mcmc_data = sample_langevin(
            x0.detach(),
            self.energy,
            n_steps=self.x_step,
            step_size=self.x_step_size,
            noise_scale=self.x_noise_std,
            temperature=self.x_temperature,
            clip=self.x_bound,
            clip_grad=self.x_clip_grad,
            spherical=False,
            mh=self.x_mh,
            reject_boundary=self.x_reject_boundary,
        )

        mcmc_data["sample_x"] = mcmc_data.pop("sample")
        if replay:
            self.buffer.push(mcmc_data["sample_x"])

        return mcmc_data

    def __sample_z(self, n_sample=None, device=None, replay=False, z0=None):
        if z0 is None:
            z0 = self.__initial_sample(n_sample, device)
        energy = lambda z: self.energy(self.decoder(z))
        mcmc_data = sample_langevin(
            z0,
            energy,
            step_size=self.z_step_size,
            n_steps=self.z_step,
            noise_scale=self.z_noise_std,
            temperature=self.temperature,
            clip=self.z_bound,
            clip_grad=self.z_clip_langevin_grad,
            spherical=self.spherical,
            mh=self.z_mh,
            reject_boundary=self.z_reject_boundary,
        )

        if replay:
            self.buffer.push(mcmc_data["sample"])
        return mcmc_data

    def __sample_omi(self, n_sample, device, replay=False):
        """using on-manifold initialization"""
        # Step 1: On-manifold initialization: LMC on Z space
        z0 = self.__initial_sample(n_sample, device)
        if self.spherical:
            z0 = self.__normalize(z0)
        d_sample_z = self.__sample_z(z0=z0, replay=replay)
        sample_z = d_sample_z["sample"]

        sample_x_1 = self.decoder(sample_z).detach()
        if self.x_bound is not None:
            sample_x_1.clamp_(self.x_bound[0], self.x_bound[1])

        # Step 2: LMC on X space
        d_sample_x = self.sample_x(x0=sample_x_1, replay=False)
        sample_x_2 = d_sample_x["sample"]
        return {
            "sample_x": sample_x_2,
            "sample_z": sample_z.detach(),
            "sample_x0": sample_x_1,
            "sample_z0": z0.detach(),
        }

    def sample(self, x0=None, n_sample=None, device=None, replay=None):
        """Sampling factory function.
        
        Takes either x0 or n_sample and device.
        """

        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        if replay is None:
            replay = self.replay

        if self.sampling == "cd":
            return self.__sample_x(n_sample, device, x0=x0, replay=False)
        elif self.sampling == "pcd":
            return self.__sample_x(n_sample, device, replay=replay)
        elif self.sampling == "omi":
            return self.__sample_omi(n_sample, device, replay=replay)

#    def __compute_emd(self, positive_samples, negative_samples):
 #       if int(ot.__version__.split(".")[1]) < 9:
  #          log.warning(f"Your optimal transport ot version is {ot.__version__}")
   #         log.warning(f"EMD calculation not supported for gradient descent, will probably crash.")
    #    loss_matrix = ot.dist(positive_samples, negative_samples)
     #   n_examples = len(positive_samples)
      #  weights = torch.ones(n_examples) / n_examples
       # emd = ot.emd2(
        #    weights,
         #   weights,
          #  loss_matrix,
           # numItermax=1e6,
        #)

        #return emd

    def __compute_emd(self, positive_samples, negative_samples):
        """
        Compute the Earth Mover's Distance (EMD) between positive_samples and negative_samples
        in a way that keeps the result inside the autograd graph.  All helper tensors are created
        on the same device/dtype as the inputs so no CUDA/CPU mismatch occurs.

        Returns
        -------
        torch.Tensor
            A scalar tensor whose `.requires_grad` follows from the inputs, so it can be used as
            a loss term in `backward()`.
        """
        # POT ≥ 0.9 supports a fully‑torch backend; keep everything on the input device
        device = positive_samples.device
        dtype  = positive_samples.dtype

        # Pairwise cost matrix (torch, differentiable)
        loss_matrix = ot.dist(positive_samples, negative_samples, p=2)

        n_examples = positive_samples.size(0)
        weights = torch.ones(n_examples, device=device, dtype=dtype) / n_examples

        emd = ot.emd2(weights, weights, loss_matrix, numItermax=1_000_000)

        return emd

    def __wnae_step(self, x, mcmc_replay=True, compute_emd=True, detach_negative_samples=False, run_mcmc=True):
        """WNAE step.
        
        Args:
            x (torch.Tensor): Data
            mcmc_replay (bool, optional, default=True): Set to True if the MCMC
                samples obtained should be added to the buffer for replay
        """

        positive_energy, positive_z = self.energy_with_z(x)
        ae_loss = positive_energy.mean()

        training_dict = {
            "reco_errors": positive_energy.detach().cpu(),
            "positive_energy": positive_energy.mean().item(),
            "positive_z": positive_z.detach().cpu(),
        }

        if run_mcmc:
            self.__set_shapes(x)
            mcmc_data = self.sample(x, replay=mcmc_replay)
            negative_samples = mcmc_data.pop("sample_x")
            if detach_negative_samples:
                negative_samples = negative_samples.detach()

            negative_energy, negative_z = self.energy_with_z(negative_samples)

            if compute_emd:
                loss = self.__compute_emd(x, negative_samples)
                training_dict["loss"] = loss.item()
            else:
                loss = None

            nae_loss = positive_energy.mean() - negative_energy.mean()

            training_dict.update({
                "negative_samples": negative_samples.detach().cpu(),
                "negative_energy": negative_energy.mean().item(),
                "negative_z": negative_z.detach().cpu(),
                "mcmc_data": mcmc_data,
            })

        else:
            loss = None
            nae_loss = None

        return loss, ae_loss, nae_loss, training_dict

    def train_step(self, x):
        """WANE training step.
        
        TODO: Write documentation!

        Args:
            x (torch.Tensor)
        
        Returns:
            Loss function and a dict[str, any]: The first element returned is the loss
            function on which to call backward. The second element returned is a dict
            with many information concerning the training:

                - "loss": the loss function
                - "positive_energy": the positive energy
        """

        loss, _, _, training_dict = self.__wnae_step(x, mcmc_replay=True)
        return loss, training_dict

    def train_step_ae(self, x, run_mcmc=False):
        """Standard AE training.
        
        TODO: Write documentation!

        Args:
            x (torch.Tensor)
        
        Returns:
            Loss function and a dict[str, any]: The first element returned is the loss
            function on which to call backward. The second element returned is a dict
            with many information concerning the training:

                - "loss": the loss function
                - "positive_energy": the positive energy
        """

        _, loss, _, training_dict = self.__wnae_step(x, mcmc_replay=True, detach_negative_samples=True, run_mcmc=run_mcmc)
        training_dict["loss"] = loss.item()  # overwrite WNAE loss by AE loss
        return loss, training_dict

    def train_step_nae(self, x):
        """Standard NAE training.
        
        TODO: Write documentation!

        Args:
            x (torch.Tensor)
        
        Returns:
            Loss function and a dict[str, any]: The first element returned is the loss
            function on which to call backward. The second element returned is a dict
            with many information concerning the training:

                - "loss": the loss function
                - "positive_energy": the positive energy
        """

        _, _, loss, training_dict = self.__wnae_step(x, mcmc_replay=True, detach_negative_samples=True)
        training_dict["loss"] = loss.item()  # overwrite WNAE loss by NAE loss
        return loss, training_dict

    def validation_step(self, x):
        """Validation step of the wnae.utils.
        
        TODO: Write documentation!
        """

        return self.__wnae_step(x, mcmc_replay=False, compute_emd=True)[3]

    def evaluate(self, x):
        """Perform WNAE evaluation.
        
        TODO: Write documentation!
        """

        return self.__wnae_step(x, mcmc_replay=False, compute_emd=False)[3]

    def run_mcmc(self, x, replay=False):
        """Run MCMC.
        
        TODO: Write documentation!
        """

        self.__set_shapes(x)
        d_sample = self.sample(x, replay=replay)
        return d_sample["sample_x"]

