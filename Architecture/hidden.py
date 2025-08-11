# hiddend_pytorch.py
# Full reference implementation of HiDDeN (Zhu et al. 2018)
# Requires: torch, torchvision, tqdm, pillow, numpy
# Usage: point DATA_ROOT to a folder of images arranged for ImageFolder
# Citation: HiDDeN: Hiding Data With Deep Networks. Zhu et al., arXiv:1807.09937. :contentReference[oaicite:1]{index=1}

import os
import math
import random
from typing import Tuple, Optional, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# --------------------------
# Utilities
# --------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# binary message utilities
def sample_random_bits(batch_size: int, L: int, device=device):
    return torch.randint(0, 2, (batch_size, L), dtype=torch.float32, device=device)

def bits_to_str(bits: torch.Tensor) -> str:
    # bits: (L,) or (B,L)
    if bits.dim() == 1:
        bits = bits.unsqueeze(0)
    return "".join(str(int(b.item())) for b in bits.view(-1))

# PSNR helper
def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(max_val**2 / mse)

# --------------------------
# Convolutional blocks
# --------------------------
def conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
        nn.BatchNorm2d(out_ch),
        activation(inplace=True)
    )

# --------------------------
# Encoder, Decoder, Adversary (paper architecture is modular; below is a practical blueprint)
# --------------------------
class Encoder(nn.Module):
    """
    Encoder E_theta: receives cover image I_co (B,C,H,W) and message bits (B,L)
    Produces encoded image I_en of same shape as I_co.
    """
    def __init__(self, in_ch=3, message_len=100, hid_channels=64):
        super().__init__()
        self.message_len = message_len
        # initial conv stack to produce spatial features
        self.front = nn.Sequential(
            conv_block(in_ch, hid_channels, 3, 1, 1),
            conv_block(hid_channels, hid_channels, 3, 1, 1),
            conv_block(hid_channels, hid_channels, 3, 1, 1),
        )
        # after replicating message we'll concatenate it -> hid_channels + message_len channels
        self.middle = nn.Sequential(
            conv_block(hid_channels + message_len, hid_channels, 3, 1, 1),
            conv_block(hid_channels, hid_channels, 3, 1, 1),
            conv_block(hid_channels, hid_channels, 3, 1, 1),
        )
        # output residual / image
        self.final = nn.Sequential(
            nn.Conv2d(hid_channels, in_ch, kernel_size=1),
            nn.Sigmoid()  # expects images normalized to [0,1]
        )

    def forward(self, img: torch.Tensor, bits: torch.Tensor):
        # img: (B,C,H,W), bits: (B,L) float in {0,1}
        B, C, H, W = img.shape
        x = self.front(img)
        # replicate message to spatial volume
        # bits -> (B, L, H, W)
        bits_vol = bits.view(B, self.message_len, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([x, bits_vol], dim=1)
        x = self.middle(x)
        out = self.final(x)
        # optionally we can output residual: img + residual clipped, but paper outputs full image
        return out

class Decoder(nn.Module):
    """
    Decoder D_phi: receives noised image Ino and outputs predicted message (B,L) floats.
    Follows conv -> global average pool -> linear
    """
    def __init__(self, in_ch=3, message_len=100, hid_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(in_ch, hid_channels, 3, 1, 1),
            conv_block(hid_channels, hid_channels, 3, 1, 1),
            conv_block(hid_channels, hid_channels, 3, 1, 1),
        )
        # final conv to produce L channels
        self.to_bits = nn.Conv2d(hid_channels, message_len, kernel_size=1)
        self.linear = nn.Linear(message_len, message_len)  # optional small linear projection

    def forward(self, ino: torch.Tensor):
        x = self.conv(ino)  # (B, hid, H, W)
        x = self.to_bits(x)  # (B, L, H, W)
        # global avg pool
        x = x.mean(dim=[2,3])  # (B, L)
        x = self.linear(x)
        # outputs are real-valued -> train with MSE towards 0/1 bits
        return x

class Discriminator(nn.Module):
    """
    Adversary A_gamma: binary classifier predicting whether image is encoded
    """
    def __init__(self, in_ch=3, hid_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(in_ch, hid_channels, 3, 1, 1),
            conv_block(hid_channels, hid_channels, 3, 1, 1),
            conv_block(hid_channels, hid_channels, 3, 1, 1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hid_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor):
        x = self.conv(img)
        return self.classifier(x).view(-1)

# --------------------------
# Noise layers (parameterless). Each maps I_en & I_co -> I_no
# --------------------------
def identity_noise(I_en, I_co, **kwargs):
    return I_en

def pixel_dropout_noise(I_en, I_co, p=0.3):
    # for each pixel, with prob p keep encoded, else replace with cover
    mask = (torch.rand_like(I_en) < p).float()
    return I_en * mask + I_co * (1 - mask)

def cropout_noise(I_en, I_co, p=0.3):
    # keep a random contiguous square region of I_en of area fraction p; outside use cover
    B, C, H, W = I_en.shape
    keep_area = int(math.sqrt(p) * H)
    if keep_area < 1:
        return I_co
    y0 = torch.randint(0, H - keep_area + 1, (1,)).item()
    x0 = torch.randint(0, W - keep_area + 1, (1,)).item()
    out = I_co.clone()
    out[:, :, y0:y0+keep_area, x0:x0+keep_area] = I_en[:, :, y0:y0+keep_area, x0:x0+keep_area]
    return out

def crop_noise(I_en, I_co, p=0.035):
    # crop a small region of I_en and feed only that crop to decoder
    B, C, H, W = I_en.shape
    keep_area = int(math.sqrt(p) * H)
    if keep_area < 1:
        # degenerate: return small central crop to avoid empty
        y0 = H//2 - 1
        x0 = W//2 - 1
        return I_en[:, :, y0:y0+2, x0:x0+2]
    y0 = torch.randint(0, H - keep_area + 1, (1,)).item()
    x0 = torch.randint(0, W - keep_area + 1, (1,)).item()
    return I_en[:, :, y0:y0+keep_area, x0:x0+keep_area]

def gaussian_blur_noise(I_en, I_co, sigma=1.0, kernel_size=9):
    # differentiable gaussian blur via conv2d
    if kernel_size % 2 == 0:
        kernel_size += 1
    B, C, H, W = I_en.shape
    # create 1D gaussian
    coords = torch.arange(kernel_size) - (kernel_size - 1) / 2.
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    # separable kernel via outer product
    kernel2d = torch.outer(g, g).to(I_en.device).float()  # (k,k)
    kernel2d = kernel2d.view(1, 1, kernel_size, kernel_size)
    kernel2d = kernel2d.repeat(C, 1, 1, 1)  # (C,1,k,k)
    # pad and conv
    pad = kernel_size // 2
    out = F.conv2d(F.pad(I_en, (pad, pad, pad, pad), mode='reflect'), kernel2d, groups=C)
    return out

# --------------------------
# Differentiable JPEG approx: DCT conv + mask/drop + inverse DCT
# --------------------------
def build_dct_kernels(block_size=8, device='cpu', dtype=torch.float32):
    """Return (block_size*block_size, 1, block_size, block_size) tensor of DCT basis kernels.
       Implementation follows standard 2D DCT basis definitions; these are fixed kernels.
    """
    N = block_size
    kernels = []
    for u in range(N):
        for v in range(N):
            # create basis function for coordinates (x,y)
            mat = np.zeros((N, N), dtype=np.float32)
            for x in range(N):
                for y in range(N):
                    mat[x, y] = (np.cos((2*x+1) * u * np.pi / (2*N)) *
                                 np.cos((2*y+1) * v * np.pi / (2*N)))
            alpha_u = np.sqrt(1.0/N) if u == 0 else np.sqrt(2.0/N)
            alpha_v = np.sqrt(1.0/N) if v == 0 else np.sqrt(2.0/N)
            mat *= (alpha_u * alpha_v)
            kernels.append(mat)
    kernels = np.stack(kernels, axis=0)  # (N*N, N, N)
    kernels = torch.tensor(kernels, dtype=dtype, device=device).unsqueeze(1)  # (K,1,N,N)
    return kernels  # apply per-channel by conv groups

def rgb_to_ycbcr(img):
    # expect img in [0,1], shape (B,3,H,W)
    # Use standard conversion matrix
    r = img[:,0:1]
    g = img[:,1:1+1]
    b = img[:,2:2+1]
    y  =  0.299*r + 0.587*g + 0.114*b
    cb = -0.168736*r - 0.331264*g + 0.5*b + 0.5
    cr =  0.5*r - 0.418688*g - 0.081312*b + 0.5
    return torch.cat([y, cb, cr], dim=1)

def ycbcr_to_rgb(img):
    y = img[:,0:1]
    cb = img[:,1:2] - 0.5
    cr = img[:,2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

class DifferentiableJPEG:
    """
    Implements two differentiable approximations used in HiDDeN:
      - jpeg_mask: zero out high-frequency DCT coefficients per block
      - jpeg_drop: probabilistic drop of coefficients with frequency-weighted probability
    The transform works on YCbCr channels and uses block convs with stride=8.
    """
    def __init__(self, block_size=8, mask_keep_low=10, device='cpu'):
        self.bs = block_size
        self.device = device
        self.kernels = build_dct_kernels(block_size, device=device)  # (K,1,bs,bs)
        self.K = self.kernels.shape[0]

    def _dct_blocks(self, img):
        # img shape (B,1,H,W). We want DCT coefficients per non-overlapping block.
        B, C, H, W = img.shape
        assert C == 1
        # conv with stride=bs and kernel size=bs, using kernels grouped by channel=1
        # We'll apply conv with in_channels=1, out_channels=K, stride=bs
        # pad if necessary so H and W divisible by bs
        pad_h = (self.bs - (H % self.bs)) % self.bs
        pad_w = (self.bs - (W % self.bs)) % self.bs
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            H += pad_h; W += pad_w
        # conv: groups=1
        # kernels on proper device
        kernels = self.kernels.to(img.device).float()  # (K,1,bs,bs)
        coeffs = F.conv2d(img, kernels, bias=None, stride=self.bs)  # (B, K, H/bs, W/bs)
        return coeffs, img.shape[2], img.shape[3]

    def _idct_from_blocks(self, coeffs, out_H, out_W):
        # coeffs: (B, K, H_blocks, W_blocks)
        # reconstruct by conv_transpose with stride=bs and kernels equal to DCT basis
        kernels = self.kernels.to(coeffs.device).float()
        # conv_transpose expects weight shape (in_channels, out_channels/groups, kH, kW)
        # We'll transpose dimensions by swapping in/out notion: treat coeff channels as in_channels
        # But easiest: use F.conv_transpose2d with kernels shaped (K,1,bs,bs) and groups=K with input grouped per channel
        # So expand coeffs per group: reshape (B, K, Hb, Wb) -> (B*K,1,Hb,Wb) and perform conv_transpose with groups=K? Simpler approach:
        # Use conv_transpose2d with weight shape (K,1,bs,bs) and groups=1 which maps K->bs*bs combination; to get correct mapping we do:
        out = F.conv_transpose2d(coeffs, kernels, stride=self.bs, padding=0)  # (B,1, out_H_pad, out_W_pad)
        # crop to desired size
        out = out[..., :out_H, :out_W]
        return out

    def jpeg_mask(self, img_ycbcr, keep_low=10):
        # img_ycbcr: (B,3,H,W)
        B, C, H, W = img_ycbcr.shape
        out_channels = []
        # Apply per-channel DCT blocks
        for ch in range(3):
            channel = img_ycbcr[:, ch:ch+1, :, :]
            coeffs, out_H, out_W = self._dct_blocks(channel)
            # zero out high-frequency coefficients: keep only the first keep_low indices (low-frequency)
            # simple strategy: zero all coeffs with flat index >= keep_low
            mask = torch.zeros_like(coeffs)
            # keep first keep_low frequencies
            K = coeffs.shape[1]
            keep_low = min(keep_low, K)
            mask[:, :keep_low, :, :] = 1.0
            coeffs = coeffs * mask
            rec = self._idct_from_blocks(coeffs, out_H, out_W)
            out_channels.append(rec)
        ycbcr_rec = torch.cat(out_channels, dim=1)
        rgb = ycbcr_to_rgb(ycbcr_rec)
        return rgb

    def jpeg_drop(self, img_ycbcr, base_drop_p=0.2):
        # Randomly drop coefficients with probability increasing with frequency index
        B, C, H, W = img_ycbcr.shape
        out_channels = []
        for ch in range(3):
            channel = img_ycbcr[:, ch:ch+1, :, :]
            coeffs, out_H, out_W = self._dct_blocks(channel)
            K = coeffs.shape[1]
            # drop probability increases linearly with index
            freqs = torch.arange(K, dtype=torch.float32, device=coeffs.device)
            norm = (freqs / (K - 1))
            drop_probs = base_drop_p * (0.1 + norm)  # scale
            # sample mask per coefficient map (B,K,Hb,Wb)
            mask = (torch.rand_like(coeffs) > drop_probs.view(1, K, 1, 1)).float()
            coeffs = coeffs * mask
            rec = self._idct_from_blocks(coeffs, out_H, out_W)
            out_channels.append(rec)
        ycbcr_rec = torch.cat(out_channels, dim=1)
        rgb = ycbcr_to_rgb(ycbcr_rec)
        return rgb

def jpeg_mask_noise(I_en, I_co, djpg: DifferentiableJPEG, keep_low=10):
    # apply approx jpeg in YCbCr domain but keep U,V maybe lower keep; here we use same for simplicity
    ycbcr = rgb_to_ycbcr(I_en)
    return djpg.jpeg_mask(ycbcr, keep_low)

def jpeg_drop_noise(I_en, I_co, djpg: DifferentiableJPEG, base_drop_p=0.2):
    ycbcr = rgb_to_ycbcr(I_en)
    return djpg.jpeg_drop(ycbcr, base_drop_p)

# --------------------------
# Training loop and helpers
# --------------------------
class HiDDeNTrainer:
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        discriminator: Discriminator,
        djpg: Optional[DifferentiableJPEG] = None,
        lr=1e-3,
        lambda_i=1.0,
        lambda_g=1e-3,
        message_len=100,
        device=device
    ):
        self.enc = encoder.to(device)
        self.dec = decoder.to(device)
        self.dis = discriminator.to(device)
        self.djpg = djpg
        self.device = device
        self.message_len = message_len
        self.lambda_i = lambda_i
        self.lambda_g = lambda_g

        # optimizers
        params_ed = list(self.enc.parameters()) + list(self.dec.parameters())
        self.opt_ed = torch.optim.Adam(params_ed, lr=lr)
        self.opt_dis = torch.optim.Adam(self.dis.parameters(), lr=lr)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def one_step(self, imgs):
        # imgs in [0,1] (B,3,H,W)
        B = imgs.shape[0]
        # sample messages
        msgs = sample_random_bits(B, self.message_len, device=self.device)  # (B,L)
        # forward encode
        i_en = self.enc(imgs, msgs)  # encoded images in [0,1]
        # choose noise type randomly from a set
        noise_choice = random.choice(['identity', 'dropout', 'cropout', 'gaussian', 'jpeg_mask', 'jpeg_drop'])
        if noise_choice == 'identity':
            i_no = identity_noise(i_en, imgs)
        elif noise_choice == 'dropout':
            i_no = pixel_dropout_noise(i_en, imgs, p=0.7)  # keep majority of encoded pixels
        elif noise_choice == 'cropout':
            i_no = cropout_noise(i_en, imgs, p=0.3)
        elif noise_choice == 'gaussian':
            i_no = gaussian_blur_noise(i_en, imgs, sigma=1.5, kernel_size=9)
        elif noise_choice == 'jpeg_mask' and (self.djpg is not None):
            i_no = jpeg_mask_noise(i_en, imgs, self.djpg, keep_low=12)
        elif noise_choice == 'jpeg_drop' and (self.djpg is not None):
            i_no = jpeg_drop_noise(i_en, imgs, self.djpg, base_drop_p=0.35)
        else:
            # fall back
            i_no = identity_noise(i_en, imgs)

        # -------------------------
        # Update discriminator (A_gamma)
        # -------------------------
        self.opt_dis.zero_grad()
        pred_real = self.dis(imgs)  # should be close to 0 (no message)
        pred_fake = self.dis(i_en.detach())  # should be close to 1 (encoded)
        # targets: real -> 0, fake -> 1 as per LA = log(1-A(Ico)) + log(A(Ien))
        target_real = torch.zeros_like(pred_real)
        target_fake = torch.ones_like(pred_fake)
        loss_A = self.bce(pred_real, target_real) + self.bce(pred_fake, target_fake)
        loss_A.backward()
        self.opt_dis.step()

        # -------------------------
        # Update encoder+decoder (theta, phi)
        # -------------------------
        self.opt_ed.zero_grad()
        # decode
        m_hat = self.dec(i_no)  # (B,L) real values
        loss_M = self.mse(m_hat, msgs)  # message distortion
        # image distortion
        loss_I = F.mse_loss(i_en, imgs)  # pixelwise L2
        # adversarial term: encourage A(i_en) small (generator loss LG = log(1-A(Ien)) in paper)
        pred_for_gen = self.dis(i_en)
        # to minimize log(1 - A(i_en)), use BCE with target 0 (i.e., make discriminator think it's real)
        gen_target = torch.zeros_like(pred_for_gen)
        loss_Gterm = self.bce(pred_for_gen, gen_target)
        # total
        loss_ED = loss_M + self.lambda_i * loss_I + self.lambda_g * loss_Gterm
        loss_ED.backward()
        self.opt_ed.step()

        # compute simple metrics
        with torch.no_grad():
            # decode bits thresholded at 0.5
            decoded_bits = (torch.sigmoid(m_hat) >= 0.5).float()
            # but note: our decoder outputs are not passed through sigmoid during training; apply sigmoid now to map to [0,1]
            bit_acc = (decoded_bits == msgs).float().mean().item()
            psnr_val = psnr(i_en, imgs).item()

        return {
            "loss_A": loss_A.item(),
            "loss_M": loss_M.item(),
            "loss_I": loss_I.item(),
            "loss_G": loss_Gterm.item(),
            "loss_ED": loss_ED.item(),
            "bit_acc": bit_acc,
            "psnr": psnr_val,
            "noise": noise_choice
        }

    def evaluate_on_batch(self, imgs, noise="identity"):
        B = imgs.shape[0]
        msgs = sample_random_bits(B, self.message_len, device=self.device)
        with torch.no_grad():
            i_en = self.enc(imgs, msgs)
            if noise == 'identity':
                i_no = identity_noise(i_en, imgs)
            elif noise == 'dropout':
                i_no = pixel_dropout_noise(i_en, imgs, p=0.7)
            elif noise == 'cropout':
                i_no = cropout_noise(i_en, imgs, p=0.3)
            elif noise == 'gaussian':
                i_no = gaussian_blur_noise(i_en, imgs, sigma=1.5, kernel_size=9)
            elif noise == 'jpeg_mask' and (self.djpg is not None):
                i_no = jpeg_mask_noise(i_en, imgs, self.djpg, keep_low=12)
            elif noise == 'jpeg_drop' and (self.djpg is not None):
                i_no = jpeg_drop_noise(i_en, imgs, self.djpg, base_drop_p=0.35)
            else:
                i_no = identity_noise(i_en, imgs)

            m_hat = self.dec(i_no)
            decoded_bits = (torch.sigmoid(m_hat) >= 0.5).float()
            bit_acc = (decoded_bits == msgs).float().mean().item()
            return {
                "bit_acc": bit_acc,
                "psnr": psnr(i_en, imgs).item()
            }

# --------------------------
# Data loader and training runner
# --------------------------
def make_dataloader(data_root, img_size=128, batch_size=12, num_workers=4):
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # to [0,1]
    ])
    ds = ImageFolder(data_root, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return dl

def train_loop(
    data_root: str,
    out_dir: str,
    epochs: int = 200,
    img_size: int = 128,
    batch_size: int = 12,
    message_len: int = 100,
    lr: float = 1e-3,
    lambda_i: float = 1.0,
    lambda_g: float = 1e-3,
    device=device
):
    os.makedirs(out_dir, exist_ok=True)
    dl = make_dataloader(data_root, img_size, batch_size)
    enc = Encoder(in_ch=3, message_len=message_len, hid_channels=64)
    dec = Decoder(in_ch=3, message_len=message_len, hid_channels=64)
    dis = Discriminator(in_ch=3, hid_channels=64)
    djpg = DifferentiableJPEG(block_size=8, device=device)

    trainer = HiDDeNTrainer(enc, dec, dis, djpg=djpg, lr=lr, lambda_i=lambda_i, lambda_g=lambda_g, message_len=message_len, device=device)

    global_step = 0
    for epoch in range(epochs):
        epoch_stats = {"loss_A":0.0, "loss_ED":0.0, "bit_acc":0.0, "psnr":0.0}
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            stats = trainer.one_step(imgs)
            global_step += 1
            for k in ["loss_A", "loss_ED", "bit_acc", "psnr"]:
                epoch_stats[k] += stats[k]
            if global_step % 200 == 0:
                # save model checkpoint periodically
                ckpt = {
                    "enc": trainer.enc.state_dict(),
                    "dec": trainer.dec.state_dict(),
                    "dis": trainer.dis.state_dict(),
                    "opt_ed": trainer.opt_ed.state_dict(),
                    "opt_dis": trainer.opt_dis.state_dict(),
                }
                torch.save(ckpt, os.path.join(out_dir, f"ckpt_step_{global_step}.pth"))
            pbar.set_postfix({"bit_acc": stats["bit_acc"], "psnr": stats["psnr"], "noise": stats["noise"]})
        # average stats
        n = len(dl)
        avg_stats = {k: epoch_stats[k]/n for k in epoch_stats}
        print(f"Epoch {epoch+1} avg: {avg_stats}")
        # optionally run evaluation on held-out subset here

    # final save
    final_ckpt = {
        "enc": trainer.enc.state_dict(),
        "dec": trainer.dec.state_dict(),
        "dis": trainer.dis.state_dict()
    }
    torch.save(final_ckpt, os.path.join(out_dir, "final_model.pth"))
    print("Training complete. Model saved to", out_dir)

# --------------------------
# Example entrypoint
# --------------------------
if __name__ == "__main__":
    # configure these
    DATA_ROOT = "/path/to/images"   # point to ImageFolder-style dataset root
    OUT_DIR = "./hiddend_out"
    train_loop(DATA_ROOT, OUT_DIR, epochs=100, img_size=128, batch_size=12, message_len=100)
