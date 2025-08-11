import argparse
import os
import glob
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils

from hidden import (
    Encoder,
    Decoder,
    Discriminator,
    DifferentiableJPEG,
    HiDDeNTrainer,
    sample_random_bits,
)


# --------------------------
# Utilities
# --------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in IMAGE_EXTS


def collect_image_paths(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for inp in inputs:
        if any(ch in inp for ch in ["*", "?", "["]):
            paths.extend(glob.glob(inp))
        elif os.path.isdir(inp):
            for root, _, files in os.walk(inp):
                for f in files:
                    fp = os.path.join(root, f)
                    if is_image_file(fp):
                        paths.append(fp)
        elif os.path.isfile(inp) and is_image_file(inp):
            paths.append(inp)
    # de-dup and sort for stability
    paths = sorted(list({os.path.abspath(p) for p in paths}))
    if len(paths) == 0:
        raise FileNotFoundError("No image files found for given inputs")
    return paths


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths: List[str], img_size: Optional[int] = None):
        self.image_paths = image_paths
        tfms = []
        if img_size is not None:
            tfms.append(T.Resize((img_size, img_size)))
        tfms.append(T.ToTensor())  # maps to [0,1]
        self.transform = T.Compose(tfms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, path


def load_models(message_len: int, device: torch.device, ckpt_path: Optional[str]):
    enc = Encoder(in_ch=3, message_len=message_len, hid_channels=64).to(device)
    dec = Decoder(in_ch=3, message_len=message_len, hid_channels=64).to(device)
    dis = Discriminator(in_ch=3, hid_channels=64).to(device)
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if "enc" in ckpt:
            enc.load_state_dict(ckpt["enc"])
        if "dec" in ckpt:
            dec.load_state_dict(ckpt["dec"])
        if "dis" in ckpt and dis is not None:
            try:
                dis.load_state_dict(ckpt["dis"])  # optional if present
            except Exception:
                pass
    return enc, dec, dis


# --------------------------
# Subcommands
# --------------------------
def cmd_demo(args):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    image_paths = collect_image_paths(args.inputs)
    ds = SimpleImageDataset(image_paths, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    enc, dec, dis = load_models(args.message_len, device, ckpt_path=None)
    djpg = DifferentiableJPEG(block_size=8, device=device)
    trainer = HiDDeNTrainer(
        enc, dec, dis, djpg=djpg, lr=args.lr, lambda_i=args.lambda_i, lambda_g=args.lambda_g,
        message_len=args.message_len, device=device
    )

    global_step = 0
    for epoch in range(args.epochs):
        for batch, _paths in dl:
            batch = batch.to(device)
            stats = trainer.one_step(batch)
            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"step {global_step} | bit_acc={stats['bit_acc']:.3f} psnr={stats['psnr']:.2f} noise={stats['noise']}")
            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    # save a grid of encoded images for quick inspection
                    i_en = trainer.enc(batch, sample_random_bits(batch.size(0), args.message_len, device=device))
                    vutils.save_image(i_en.cpu(), os.path.join(args.out_dir, f"samples_step_{global_step}.png"), nrow=min(4, args.batch_size))
        if args.max_steps and global_step >= args.max_steps:
            break

    # save checkpoint
    ckpt = {
        "enc": trainer.enc.state_dict(),
        "dec": trainer.dec.state_dict(),
        "dis": trainer.dis.state_dict(),
    }
    ckpt_path = os.path.join(args.out_dir, "demo_final.pth")
    torch.save(ckpt, ckpt_path)
    print("Saved demo checkpoint:", ckpt_path)


def parse_message(args) -> torch.Tensor:
    if args.message is not None:
        msg_str = args.message.strip()
        if not set(msg_str).issubset({"0", "1"}):
            raise ValueError("--message must be a bitstring like 010101")
        bits = torch.tensor([int(c) for c in msg_str], dtype=torch.float32).unsqueeze(0)
        return bits
    elif args.message_file is not None:
        with open(args.message_file, "r") as f:
            msg_str = f.read().strip()
        if not set(msg_str).issubset({"0", "1"}):
            raise ValueError("message file must contain a bitstring like 010101")
        bits = torch.tensor([int(c) for c in msg_str], dtype=torch.float32).unsqueeze(0)
        return bits
    else:
        # random based on message_len
        return None


def cmd_encode(args):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    image_paths = collect_image_paths(args.inputs)
    ds = SimpleImageDataset(image_paths, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    enc, dec, _ = load_models(args.message_len, device, ckpt_path=args.checkpoint)
    enc.eval(); dec.eval()

    fixed_bits = parse_message(args)
    for img, path in dl:
        img = img.to(device)
        if fixed_bits is None:
            bits = sample_random_bits(1, args.message_len, device=device)
        else:
            if fixed_bits.size(1) != args.message_len:
                raise ValueError("Provided message length does not match --message-len")
            bits = fixed_bits.to(device)
        with torch.no_grad():
            i_en = enc(img, bits)
        base = os.path.splitext(os.path.basename(path[0]))[0]
        out_path = os.path.join(args.out_dir, f"{base}_encoded.png")
        vutils.save_image(i_en.cpu(), out_path)
        print(f"Saved: {out_path}")

        if args.decode:
            with torch.no_grad():
                m_hat = dec(i_en)
                m_prob = torch.sigmoid(m_hat)
                m_bits = (m_prob >= 0.5).float()
            if fixed_bits is None:
                print("Predicted bits:", "".join(str(int(b)) for b in m_bits.view(-1)))
            else:
                acc = (m_bits.cpu() == bits.cpu()).float().mean().item()
                print(f"Decode bit-acc: {acc:.3f}")


def cmd_decode(args):
    device = torch.device(args.device)
    image_paths = collect_image_paths(args.inputs)
    ds = SimpleImageDataset(image_paths, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    _, dec, _ = load_models(args.message_len, device, ckpt_path=args.checkpoint)
    dec.eval()

    gt_bits = parse_message(args)
    for img, path in dl:
        img = img.to(device)
        with torch.no_grad():
            m_hat = dec(img)
            m_prob = torch.sigmoid(m_hat)
            m_bits = (m_prob >= 0.5).float()
        bit_str = "".join(str(int(b)) for b in m_bits.view(-1))
        if gt_bits is not None:
            acc = (m_bits.cpu() == gt_bits[:, : m_bits.size(1)]).float().mean().item()
            print(f"{path[0]} | pred={bit_str} | acc={acc:.3f}")
        else:
            print(f"{path[0]} | pred={bit_str}")


def cmd_eval(args):
    device = torch.device(args.device)
    image_paths = collect_image_paths(args.inputs)
    ds = SimpleImageDataset(image_paths, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    enc, dec, dis = load_models(args.message_len, device, ckpt_path=args.checkpoint)
    djpg = DifferentiableJPEG(block_size=8, device=device)
    trainer = HiDDeNTrainer(enc, dec, dis, djpg=djpg, message_len=args.message_len, device=device)
    enc.eval(); dec.eval()

    noises = ["identity", "dropout", "cropout", "gaussian", "jpeg_mask", "jpeg_drop"]
    for noise in noises:
        bit_acc_sum = 0.0
        psnr_sum = 0.0
        count = 0
        for imgs, _paths in dl:
            imgs = imgs.to(device)
            stats = trainer.evaluate_on_batch(imgs, noise=noise)
            bit_acc_sum += stats["bit_acc"]
            psnr_sum += stats["psnr"]
            count += 1
        if count > 0:
            print(f"noise={noise:>9} | bit_acc={bit_acc_sum/count:.3f} | psnr={psnr_sum/count:.2f}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="HiDDeN runner (encode/decode/train/eval)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # demo: quick training on a handful of images
    pd = sub.add_parser("demo", help="Quick overfit training on a few images (CPU-friendly)")
    pd.add_argument("inputs", nargs="+", help="Image paths, folders or glob patterns")
    pd.add_argument("--out-dir", default="./hidden_demo_out")
    pd.add_argument("--img-size", type=int, default=128)
    pd.add_argument("--batch-size", type=int, default=8)
    pd.add_argument("--message-len", type=int, default=100)
    pd.add_argument("--epochs", type=int, default=1)
    pd.add_argument("--max-steps", type=int, default=500)
    pd.add_argument("--lr", type=float, default=1e-3)
    pd.add_argument("--lambda-i", type=float, default=1.0)
    pd.add_argument("--lambda-g", type=float, default=1e-3)
    pd.add_argument("--log-interval", type=int, default=20)
    pd.add_argument("--sample-interval", type=int, default=100)
    pd.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pd.set_defaults(func=cmd_demo)

    # encode: embed message into images
    pe = sub.add_parser("encode", help="Encode a message into images and save stego images")
    pe.add_argument("inputs", nargs="+", help="Image paths, folders or glob patterns")
    pe.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth)")
    pe.add_argument("--out-dir", default="./encoded_out")
    pe.add_argument("--img-size", type=int, default=None, help="Optional resize square size")
    pe.add_argument("--message-len", type=int, default=100)
    pe.add_argument("--message", type=str, default=None, help="Bitstring like 010101... If omitted, random per image")
    pe.add_argument("--message-file", type=str, default=None, help="File containing a bitstring")
    pe.add_argument("--decode", action="store_true", help="Also decode immediately and report bit-acc")
    pe.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pe.set_defaults(func=cmd_encode)

    # decode: read message from encoded images
    pd2 = sub.add_parser("decode", help="Decode message from images")
    pd2.add_argument("inputs", nargs="+", help="Image paths, folders or glob patterns")
    pd2.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    pd2.add_argument("--img-size", type=int, default=None)
    pd2.add_argument("--message-len", type=int, default=100)
    pd2.add_argument("--message", type=str, default=None, help="Optional ground-truth bitstring to compute accuracy")
    pd2.add_argument("--message-file", type=str, default=None, help="File containing ground-truth bitstring")
    pd2.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pd2.set_defaults(func=cmd_decode)

    # eval: robustness on noise layers
    pv = sub.add_parser("eval", help="Evaluate robustness across noise layers")
    pv.add_argument("inputs", nargs="+", help="Image paths, folders or glob patterns")
    pv.add_argument("--checkpoint", type=str, required=True)
    pv.add_argument("--img-size", type=int, default=128)
    pv.add_argument("--message-len", type=int, default=100)
    pv.add_argument("--batch-size", type=int, default=8)
    pv.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pv.set_defaults(func=cmd_eval)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


