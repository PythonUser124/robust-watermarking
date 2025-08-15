"""
latent_watermark_architecture.py
===============================

The architecture is broken down into the following conceptual pieces:

* **Stable Diffusion Wrapper** – An external module (for example
  implemented using `diffusers`) responsible for sampling latent noise
  and producing images.  In our context the UNet and VAE are kept
  frozen; only the watermark encoders/decoders are trainable.
* **BinScheduler** – Handles selection of a fixed number of Fourier
  coefficients (bins) in a mid–frequency annulus for each diffusion
  step based on a deployment key.  The annulus helps ensure the
  watermark is imperceptible while still surviving JPEG and resizing.
* **EncoderMLP / DecoderMLP** – Simple multi–layer perceptron classes
  operating over the selected bins.  The encoder maps a small bit
  string plus a time embedding into amplitude offsets, while the
  decoder maps FFT magnitudes back into bit predictions.
* **LatentWatermarkModel** – A façade that ties the above modules
  together.  It exposes high level `embed` and `decode` functions
  alongside stubs for loss computation, training, augmentation and
  interpretability.  Users of this class are expected to implement
  inversion and training logic outside this file.


"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Optional external dependencies
###############################################################################

# The classes and functions in this module are written to be as self–contained
# as possible.  To actually perform sampling or inversion with Stable Diffusion
# one should depend on the HuggingFace `diffusers` library.  However, this
# import is optional and only used within the `StableDiffusionFrozen` wrapper
# below.  If `diffusers` is not available at runtime, that wrapper will
# gracefully raise an informative error when instantiated.  Likewise the
# interpretability utilities optionally rely on torchvision.models (e.g. VGG).
try:
    from diffusers import StableDiffusionPipeline  # type: ignore
except ImportError:
    StableDiffusionPipeline = None  # fallback for environments without diffusers

try:
    import torchvision.models as tv_models  # type: ignore
except ImportError:
    tv_models = None  # fallback if torchvision is unavailable

###############################################################################
# Helper classes
###############################################################################

# NOTE: The BinScheduler class has been removed to disable bin scheduling.


class EncoderMLP(nn.Module):
    """Small multi–layer perceptron to generate bin offsets from bits and time.

    Parameters
    ----------
    bits_per_step : int
        Number of payload bits embedded at each selected diffusion step.
    M : int
        Number of FFT bins per step (must match the value used in
        `BinScheduler`).
    hidden : int
        Width of the hidden layers; default 256 is typically sufficient.
    time_embed_dim : int
        Dimensionality of the sinusoidal time embedding appended to the
        bit vector.  Set to 0 to omit time embedding.
    """

    def __init__(self, bits_per_step: int, M: int, hidden: int = 256, time_embed_dim: int = 1):
        super().__init__()
        self.bits_per_step = bits_per_step
        self.M = M
        input_dim = bits_per_step + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, M),
        )
        self.time_embed_dim = time_embed_dim

    def forward(self, bits: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """Map bits and time embedding to amplitude offsets.

        Parameters
        ----------
        bits : torch.Tensor
            Tensor of shape `(B, bits_per_step)` with values 0/1 (float).
        t_embed : torch.Tensor
            Tensor of shape `(B, time_embed_dim)` providing a normalised
            representation of the diffusion step (e.g. step index divided by
            total steps or a sinusoidal encoding).

        Returns
        -------
        torch.Tensor
            Tensor of shape `(B, M)` representing the per–bin amplitude
            offsets.  Users are expected to broadcast this to the latent
            channel dimension when applying to complex FFT magnitudes.
        """
        if self.time_embed_dim == 0:
            x = bits
        else:
            x = torch.cat([bits, t_embed], dim=-1)
        out = self.net(x)
        return out


class DecoderMLP(nn.Module):
    """Small MLP to map FFT magnitudes back to payload bits per step.

    Parameters
    ----------
    M : int
        Number of magnitudes per step (must match the value used in
        `BinScheduler`).
    bits_per_step : int
        Number of payload bits embedded at each selected diffusion step.
    hidden : int
        Width of hidden layers; default 256.
    time_embed_dim : int
        Dimensionality of the time embedding concatenated to the input; same
        as in `EncoderMLP`.
    """

    def __init__(self, M: int, bits_per_step: int, hidden: int = 256, time_embed_dim: int = 1):
        super().__init__()
        self.M = M
        self.bits_per_step = bits_per_step
        input_dim = M + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bits_per_step),
        )
        self.time_embed_dim = time_embed_dim

    def forward(self, mags: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """Predict bits from magnitudes and time embedding.

        Parameters
        ----------
        mags : torch.Tensor
            Tensor of shape `(B, M)` representing the averaged FFT
            magnitudes of the selected bins across channels.
        t_embed : torch.Tensor
            Tensor of shape `(B, time_embed_dim)` representing the
            normalised diffusion step.
        Returns
        -------
        torch.Tensor
            Tensor of shape `(B, bits_per_step)` containing logits for
            each embedded bit.  Note that outputs are real–valued and
            should be passed through a sigmoid or BCE loss as appropriate.
        """
        if self.time_embed_dim == 0:
            x = mags
        else:
            x = torch.cat([mags, t_embed], dim=-1)
        out = self.net(x)
        return out


###############################################################################
# Core architecture
###############################################################################

@dataclass
class LatentWatermarkConfig:
    """Hyperparameters controlling the watermark architecture.

    Fields
    ------
    total_steps : int
        Total number of diffusion steps in the sampler (e.g. 50 for
        DDIM).  Anchor steps will be chosen relative to this value.
    anchor_steps : Iterable[int]
        Indices of the timesteps at which watermarking occurs.  These
        should be sorted and lie in `[0, total_steps)`.  It is often
        beneficial to choose steps uniformly in log–SNR space.
    bits_per_step : int
        Number of payload bits to embed at each anchor step.  Combined
        with the number of anchor steps this determines the payload size
        prior to ECC and sync overhead.
    M : int
        Number of bins per step as used by ``BinScheduler``.
    r_min : float
        Inner radius of the annulus relative to Nyquist.  Values in
        `[0.2, 0.3]` work well.
    r_max : float
        Outer radius of the annulus.  Choose `<0.5` to avoid fragile
        high frequencies.
    time_embed_dim : int
        Dimensionality of the time embedding; set to 0 to disable.
    """

    total_steps: int = 50
    anchor_steps: Iterable[int] = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    bits_per_step: int = 8
    M: int = 128
    r_min: float = 0.25
    r_max: float = 0.45
    time_embed_dim: int = 1


class LatentWatermarkModel(nn.Module):
    """Facade for embedding and decoding watermarks in latent diffusion models.

    This class glues together a pre–trained diffusion pipeline (e.g.
    Stable Diffusion), a `BinScheduler`, an `EncoderMLP` and a
    `DecoderMLP`.  It does not itself implement sampling, inversion or
    optimisation but instead exposes hooks for these operations.  During
    training one would drive the diffusion sampler externally, calling
    ``encode_step`` at each anchor index to insert watermark bits into
    the latent, and later using ``decode_from_image`` on the generated
    image to recover the embedded payload.

    Parameters
    ----------
    latent_size : Tuple[int, int]
        Spatial shape `(H, W)` of the latent features; for SD 1.5 this
        is `(64, 64)`.
    channels : int
        Number of channels in the latent features (e.g. 4 for SD 1.5).
    config : LatentWatermarkConfig
        Hyperparameter configuration specifying anchor steps, bits per
        step and annulus sizes.
    key : bytes
        Deployment key used to initialise the bin scheduler.
    device : Optional[torch.device]
        The device on which all modules will reside.
    """

    def __init__(
        self,
        latent_size: Tuple[int, int] = (64, 64),
        channels: int = 4,
        config: LatentWatermarkConfig | None = None,
        *,
        key: bytes,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        H, W = latent_size
        self.channels = channels
        self.config = config or LatentWatermarkConfig()
        self.device = device or torch.device("cpu")
        # Initialise scheduler and MLPs
        # BinScheduler disabled
        self.scheduler = None
        self.encoder_mlp = EncoderMLP(
            bits_per_step=self.config.bits_per_step,
            M=self.config.M,
            hidden=256,
            time_embed_dim=self.config.time_embed_dim,
        )
        self.decoder_mlp = DecoderMLP(
            M=self.config.M,
            bits_per_step=self.config.bits_per_step,
            hidden=256,
            time_embed_dim=self.config.time_embed_dim,
        )
        # Register modules on chosen device
        self.encoder_mlp.to(self.device)
        self.decoder_mlp.to(self.device)

    # ---------------------------------------------------------------------
    # Embedding API
    # ---------------------------------------------------------------------
    def time_embedding(self, step_idx: int) -> torch.Tensor:
        """Compute a simple scalar embedding for the diffusion step.

        For now we use a normalised linear scalar ``step_idx / total_steps``.
        More sophisticated sinusoidal embeddings could be substituted.
        """
        if self.config.time_embed_dim == 0:
            return torch.empty(0, device=self.device)
        # shape (1, time_embed_dim)
        return torch.tensor(
            [[step_idx / float(self.config.total_steps)] * self.config.time_embed_dim],
            dtype=torch.float32,
            device=self.device,
        )

    def encode_step(
        self,
        z_t: torch.Tensor,
        bits_step: torch.Tensor,
        step_idx: int,
        *,
        alpha_t: float = 1.0,
    ) -> torch.Tensor:
        """Inject payload bits into a latent at a given diffusion step.

        Parameters
        ----------
        z_t : torch.Tensor
            Latent tensor of shape `(B, C, H, W)` at diffusion step ``t``.
        bits_step : torch.Tensor
            Bits to embed for this step of shape `(B, bits_per_step)`.
        step_idx : int
            Index of the diffusion step (0–based, increasing as sampling
            progresses).  Should be one of the configured anchor steps.
        alpha_t : float, optional
            Scale factor controlling the energy of the perturbation.  One
            common strategy is to set `alpha_t ∝ sqrt(1 - \bar α_t)` to
            compensate for decreasing noise.

        Returns
        -------
        torch.Tensor
            Modified latent tensor of shape `(B, C, H, W)` with the
            watermark applied on the selected bins.
        """
        # Determine bins and PN sequence for this step
        bins = self.scheduler.bins_for_step(step_idx).to(self.device)  # (M,2)
        pn = self.scheduler.pn_for_step(step_idx).to(self.device)  # (M)
        B, C, H, W = z_t.shape
        # Compute time embedding and encode amplitude offsets
        t_embed = self.time_embedding(step_idx).expand(B, -1)  # (B, time_dim)
        delta = self.encoder_mlp(bits_step, t_embed)  # (B, M)
        # Broadcast delta and PN across channels
        delta = delta.view(B, 1, -1).repeat(1, C, 1)  # (B, C, M)
        pn = pn.view(1, 1, -1)  # (1,1,M)
        # Compute FFT, modify magnitudes, and inverse transform
        Z = torch.fft.fft2(z_t)  # complex (B,C,H,W)
        Z_shifted = torch.fft.fftshift(Z, dim=(-2, -1))  # centre DC at centre
        mags = Z_shifted.abs()
        phases = torch.angle(Z_shifted)
        y, x = bins[:, 0], bins[:, 1]
        # update magnitudes symmetrically for each selected bin and its
        # conjugate partner at (-u,-v)
        cy, cx = (H - 1) // 2, (W - 1) // 2
        y2 = (2 * cy - y) % H
        x2 = (2 * cx - x) % W
        mags[:, :, y, x] = mags[:, :, y, x] + alpha_t * delta * pn
        mags[:, :, y2, x2] = mags[:, :, y2, x2] + alpha_t * delta * pn
        # Reconstruct complex spectrum and invert
        real = mags * torch.cos(phases)
        imag = mags * torch.sin(phases)
        Z_new_shifted = torch.complex(real, imag)
        Z_new = torch.fft.ifftshift(Z_new_shifted, dim=(-2, -1))
        z_new = torch.fft.ifft2(Z_new).real
        return z_new

    # ---------------------------------------------------------------------
    # Decoding API
    # ---------------------------------------------------------------------
    def decode_from_latents(
        self, z_dict: dict[int, torch.Tensor], *, aggregate: str = "median"
    ) -> torch.Tensor:
        """Decode payload bits from a dictionary of inverted latents.

        Parameters
        ----------
        z_dict : dict[int, torch.Tensor]
            Mapping from diffusion step index to latent estimate tensor of
            shape `(B, C, H, W)`.  Only entries corresponding to
            `anchor_steps` are required.  A robust pipeline may run
            several inversion algorithms and then aggregate the results.
        aggregate : {"median", "mean", "none"}
            Strategy to combine multiple inversions before decoding.  If
            `median`, the median across inversion runs will be used.  If
            `mean`, the arithmetic mean will be used.  If `none`, the
            entries in `z_dict` are assumed to already be aggregated.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(B, total_bits)` with logits corresponding to
            all payload bits across all anchor steps.
        """
        # Prepare list of predicted logits per step
        logits_list: List[torch.Tensor] = []
        for step_idx in self.config.anchor_steps:
            if step_idx not in z_dict:
                raise KeyError(f"Missing inversion for step {step_idx}")
            z_candidates = z_dict[step_idx]  # could be (B,C,H,W) or (K,B,C,H,W)
            # If multiple inversion runs exist, reduce
            if z_candidates.dim() == 5:  # (K,B,C,H,W)
                if aggregate == "median":
                    z_use = z_candidates.median(dim=0).values
                elif aggregate == "mean":
                    z_use = z_candidates.mean(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregate}")
            else:
                z_use = z_candidates
            # Compute FFT and extract magnitudes
            Z = torch.fft.fft2(z_use)
            Z_shifted = torch.fft.fftshift(Z, dim=(-2, -1))
            mags = Z_shifted.abs()
            bins = self.scheduler.bins_for_step(step_idx).to(mags.device)
            y, x = bins[:, 0], bins[:, 1]
            # average magnitudes across channels
            mag_slice = mags[:, :, y, x].mean(dim=1)  # (B, M)
            # time embedding for this step
            B_batch = mag_slice.shape[0]
            t_embed = self.time_embedding(step_idx).expand(B_batch, -1)
            # decode
            logits_step = self.decoder_mlp(mag_slice, t_embed)
            logits_list.append(logits_step)
        # Concatenate all steps into a single tensor (B, total_bits)
        return torch.cat(logits_list, dim=-1)

    # ---------------------------------------------------------------------
    # Placeholders for loss functions and training utilities
    # ---------------------------------------------------------------------
    def compute_loss(
        self,
        logits: torch.Tensor,
        target_bits: torch.Tensor,
        *,
        loss_fn: Optional[nn.Module] = None,
        lambda_img: float = 0.05,
        perceptual_fn: Optional[nn.Module] = None,
        img_wm: Optional[torch.Tensor] = None,
        img_clean: Optional[torch.Tensor] = None,
        delta_regs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute a composite loss given predicted logits and ground truth bits.

        This convenience method allows the caller to supply a custom
        bitwise loss, perceptual loss and regularisation terms.  It
        demonstrates how losses could be combined but leaves all
        arguments optional so as not to constrain external training
        pipelines.

        Parameters
        ----------
        logits : torch.Tensor
            Raw decoder outputs of shape `(B, total_bits)`.
        target_bits : torch.Tensor
            Ground truth bit tensor of shape `(B, total_bits)`.
        loss_fn : nn.Module, optional
            A loss function mapping decoder logits to bitwise targets.
            Defaults to `nn.BCEWithLogitsLoss`.
        lambda_img : float, optional
            Weight for the perceptual loss term.  Lower values reduce
            visible artefacts at the cost of embedding strength.
        perceptual_fn : nn.Module, optional
            A differentiable perceptual distance (e.g. LPIPS) taking
            `(img_wm, img_clean) → scalar`.  If provided both
            `img_wm` and `img_clean` must be supplied.
        img_wm : torch.Tensor, optional
            Watermarked image; used only when `perceptual_fn` is not
            `None`.
        img_clean : torch.Tensor, optional
            Original image without watermark; used only when
            `perceptual_fn` is not `None`.
        delta_regs : List[torch.Tensor], optional
            List of per–step amplitude tensors used during embedding.  A
            simple L2 regulariser is applied to each; the list should be
            collected outside the model and passed in here.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor suitable for backpropagation.
        """
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()
        bit_loss = loss_fn(logits, target_bits)
        total = bit_loss
        # Optionally include perceptual/regularisation losses
        if perceptual_fn is not None and img_wm is not None and img_clean is not None:
            img_loss = perceptual_fn(img_wm, img_clean)
            total = total + lambda_img * img_loss
        if delta_regs is not None:
            reg_loss = sum(d.pow(2).mean() for d in delta_regs)
            total = total + 1e-4 * reg_loss
        return total

    # ---------------------------------------------------------------------
    # Interpretability hooks
    # ---------------------------------------------------------------------
    def perturb_vgg_layers(self, model: nn.Module, layers: List[str]) -> None:
        """Injects the encoder MLP at specified VGG layers for analysis.

        This function is a stub illustrating where one might explore
        perturbations of intermediate features.  It accepts a VGG model
        and a list of layer names/keys.  During forward passes one
        could insert watermark perturbations in these layers to study
        how downstream activations respond.  Implementation specifics
        depend on the chosen interpretability study and are beyond the
        scope of this architecture module.
        """
        raise NotImplementedError(
            "Interpretability analysis must be implemented externally."
        )


###############################################################################
# Additional modules for training, augmentation and interpretability
###############################################################################

class StableDiffusionFrozen:
    """Thin wrapper around a Stable Diffusion pipeline with frozen weights.

    This helper class encapsulates a diffusers pipeline and disables all
    gradient computation for the underlying UNet, VAE and text encoder.  It
    exposes methods for sampling images given text prompts and latent noise,
    and for inverting images back into latents (via DDIM inversion).  The
    inversion implementation is provided as a stub; users should supply
    their own inversion function appropriate to their scheduler.

    Parameters
    ----------
    pipeline : StableDiffusionPipeline
        A pre–initialised diffusers pipeline.  Must implement the usual
        `__call__`, `vae`, `text_encoder` and `unet` interfaces.
    device : torch.device, optional
        Device on which to operate.  Defaults to the pipeline's device.

    Notes
    -----
    Importing this class requires the `diffusers` package.  If it is not
    available a RuntimeError will be thrown when attempting to instantiate
    the wrapper.
    """

    def __init__(self, pipeline: "StableDiffusionPipeline", *, device: Optional[torch.device] = None):
        if StableDiffusionPipeline is None or pipeline is None:
            raise RuntimeError(
                "StableDiffusionFrozen requires diffusers; please install diffusers to use this class."
            )
        self.pipe = pipeline
        self.device = device or pipeline.device
        # Freeze all model parameters
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        # Put on device
        self.pipe = self.pipe.to(self.device)

    def sample(self, prompt: str, *, num_steps: int = 50, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate an image conditioned on a text prompt.

        This method runs the diffusion sampler with gradients disabled and
        returns the resulting image tensor of shape `(1,3,H',W')` where
        `H'` and `W'` are the decoder output resolution (e.g. 512×512).  The
        latent sampling loop can be extended by calling `encode_step` on a
        `LatentWatermarkModel` at the desired anchor steps.
        """
        with torch.no_grad():
            out = self.pipe(prompt, num_inference_steps=num_steps, generator=generator)
        return out.images[0]

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Return the text encoder hidden states for a given prompt."""
        text_inputs = self.pipe.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.pipe.text_encoder(**text_inputs).last_hidden_state

    def invert(self, image: torch.Tensor, *, num_steps: int = 50) -> dict[int, torch.Tensor]:
        """Approximate inversion of a generated image to latent representations.

        This stub shows how one could invert an image back to latent
        diffusion steps using a DDIM inversion or similar algorithm.  The
        implementation details are project–specific and should be provided
        externally.  Here we simply raise NotImplementedError.
        """
        raise NotImplementedError(
            "StableDiffusionFrozen.invert must be implemented with a suitable inversion routine."
        )


class HiddenEncoderWrapper(nn.Module):
    """Wrapper to adapt a HiDDeN encoder into a latent MLP input.

    HiDDeN encoders are convolutional networks designed to embed bits into
    pixel images.  To reuse their learned weights as an initialisation for
    our EncoderMLP, this wrapper flattens the final convolutional outputs
    and maps them to the required number of bin offsets.  In practice one
    could load a pre–trained HiDDeN encoder and train a small adapter
    layer on top to output amplitude offsets for the selected Fourier bins.

    Parameters
    ----------
    hidden_encoder : nn.Module
        A pre–trained HiDDeN encoder model.  Should take an image tensor and
        output a feature map.  Must not include the final convolution
        projecting to bits directly.
    M : int
        Number of FFT bins per step.  The adapter will map the flattened
        hidden features to this dimension.
    """

    def __init__(self, hidden_encoder: nn.Module, M: int):
        super().__init__()
        self.hidden_encoder = hidden_encoder
        self.flatten = nn.Flatten()
        # Determine output size by a dummy forward (requires example input)
        self._out_dim: Optional[int] = None
        self.adapter: Optional[nn.Linear] = None
        self.M = M

    def build(self, example_input: torch.Tensor) -> None:
        """Build the adapter layer by inferring the flattened feature dimension."""
        with torch.no_grad():
            feat = self.hidden_encoder(example_input)
            flat = self.flatten(feat)
        self._out_dim = flat.shape[-1]
        self.adapter = nn.Linear(self._out_dim, self.M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter is None:
            raise RuntimeError("HiddenEncoderWrapper.build must be called before forward")
        feat = self.hidden_encoder(x)
        flat = self.flatten(feat)
        return self.adapter(flat)


class TrainingUtils:
    """Collection of static methods to facilitate training and testing.

    This class previously provided routines for image augmentation, unit tests
    verifying tensor shapes/gradients and utilities for hyperparameter
    configuration.  Image augmentation has been intentionally removed in
    accordance with project requirements.  Unit tests and hyperparameter
    helpers remain available.
    """

    # NOTE: The augment_image function has been removed because the current
    # project does not require any data augmentation.  If future use cases
    # need augmentation, implement them externally or reintroduce a suitable
    # method.  Returning the input unchanged here ensures backward
    # compatibility with any existing training scripts that might call
    # TrainingUtils.augment_image.
    
    @staticmethod
    def augment_image(img: torch.Tensor) -> torch.Tensor:
        """Return the image unchanged.

        Augmentation functionality has been disabled.  This stub is kept to
        maintain API compatibility.  If you need data augmentation,
        implement it outside of this module or extend this method as needed.

        Parameters
        ----------
        img : torch.Tensor
            A batch of images of shape `(B, C, H, W)`.

        Returns
        -------
        torch.Tensor
            The input image tensor, unmodified.
        """
        return img

    @staticmethod
    def unit_test_shapes(model: LatentWatermarkModel, batch_size: int = 2) -> None:
        """Run simple shape checks to catch obvious bugs.

        This method constructs random latents and bit inputs, applies the
        embedding and decoding passes and asserts that output shapes match
        expectations.  It will raise AssertionError if any dimension
        mismatches occur.
        """
        H, W = model.scheduler.H, model.scheduler.W
        C = model.channels
        z = torch.randn(batch_size, C, H, W, device=model.device)
        # For each anchor step embed bits and collect modified latents
        modified = z.clone()
        delta_regs: List[torch.Tensor] = []
        for t in model.config.anchor_steps:
            bits_step = torch.randint(0, 2, (batch_size, model.config.bits_per_step), device=model.device).float()
            modified = model.encode_step(modified, bits_step, t)
        # Fake inversion dictionary with single latent per step (no ensemble)
        z_dict = {t: modified for t in model.config.anchor_steps}
        logits = model.decode_from_latents(z_dict)
        expected_bits = len(model.config.anchor_steps) * model.config.bits_per_step
        assert logits.shape == (batch_size, expected_bits), f"Expected logits shape {(batch_size, expected_bits)}, got {logits.shape}"

    @staticmethod
    def select_hyperparams(search_space: dict[str, Iterable]) -> dict[str, float]:
        """Return a simple hyperparameter configuration.

        This placeholder chooses the first element of each iterable in the
        provided search space.  In a real project one would implement a
        more sophisticated strategy (grid search, random search or Bayesian
        optimisation).  The returned dictionary can be passed to
        `LatentWatermarkConfig` or training loops.
        """
        return {k: next(iter(v)) for k, v in search_space.items()}


class InterpretabilityUtils:
    """Utilities for analysing the effects of watermark perturbations.

    These methods are intended for exploratory use to understand how
    injecting a watermark into different parts of a network changes
    downstream activations.  They are provided as stubs; concrete
    implementations should be developed based on specific interpretability
    questions.
    """

    @staticmethod
    def register_hooks(model: nn.Module, layers: List[str], hook_fn) -> List:
        """Register forward hooks on the specified layer names and return handles."""
        handles = []
        for name, module in model.named_modules():
            if name in layers:
                handles.append(module.register_forward_hook(hook_fn))
        return handles

    @staticmethod
    def inject_perturbation(features: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Add a perturbation delta to a feature tensor for analysis."""
        return features + delta

    @staticmethod
    def remove_hooks(handles: List) -> None:
        """Clean up registered hooks."""
        for h in handles:
            h.remove()


def inject_mlp_on_vgg(model: nn.Module, encoder_mlp: EncoderMLP, layers: List[str]) -> None:
    """Example of injecting an MLP perturbation into specified VGG layers.

    Given a VGG model and our encoder MLP, this function registers
    forward hooks that add small perturbations computed from the MLP to
    activations of chosen layers.  This demonstrates how one might
    study the influence of watermark embeddings on deep features.  The
    MLP is called on a dummy input (e.g. zeros) to generate the
    perturbation; one could instead compute the MLP output based on
    intermediate data.

    Parameters
    ----------
    model : nn.Module
        A pre–trained VGG network (or similar) on which to register hooks.
    encoder_mlp : EncoderMLP
        The MLP used to generate small perturbations.
    layers : List[str]
        Names of layers within the VGG model to be perturbed.
    """
    if tv_models is None:
        raise RuntimeError(
            "torchvision is required for VGG interpretability; please install torchvision."
        )
    # Create a dummy input for the MLP; here we use zeros matching bits_per_step
    dummy_bits = torch.zeros(1, encoder_mlp.bits_per_step, device=next(encoder_mlp.parameters()).device)
    dummy_time = torch.zeros(1, encoder_mlp.time_embed_dim, device=next(encoder_mlp.parameters()).device)
    perturb = encoder_mlp(dummy_bits, dummy_time)  # (1, M)
    # Flatten perturb to a scalar bias (for demonstration)
    delta = perturb.mean().item()

    def hook_fn(module: nn.Module, inputs, outputs):
        # Add a small bias to the activation
        return outputs + delta

    handles = []
    for name, layer in model.named_modules():
        if name in layers:
            handles.append(layer.register_forward_hook(hook_fn))
    # return handles for removal outside (optional), omitted for brevity
    return None


__all__ = [
    # "BinScheduler",
    "EncoderMLP",
    "DecoderMLP",
    "LatentWatermarkConfig",
    "LatentWatermarkModel",
    "StableDiffusionFrozen",
    "HiddenEncoderWrapper",
    "TrainingUtils",
    "InterpretabilityUtils",
    "inject_mlp_on_vgg",
]
