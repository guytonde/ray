import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 32, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ChannelAttention(channels),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class UpsampleNN(nn.Module):
    def __init__(self, upscale_factor: int = 2, num_res_blocks: int = 16, channels: int = 32):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, res_scale=0.1) for _ in range(num_res_blocks)]
        )
        self.conv_post_res = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_pre_upsample = nn.Conv2d(
            channels, channels * (upscale_factor ** 2), kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv_output = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        bicubic_up = F.interpolate(
            x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )
        feat = self.conv_input(x)
        res = self.res_blocks(feat)
        res = self.conv_post_res(res)
        feat = feat + res
        feat = self.conv_pre_upsample(feat)
        feat = self.pixel_shuffle(feat)
        residual_out = self.conv_output(feat)
        return bicubic_up + residual_out


def load_state_dict_compatible(model: nn.Module, model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)

    # Support either raw state_dict or wrapped checkpoint dict.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip optional DataParallel prefix.
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def image_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    # [H, W, C] uint8 -> [1, C, H, W] float32 in [0, 1]
    data = torch.from_numpy(np.array(img, dtype="float32") / 255.0).permute(2, 0, 1).unsqueeze(0)
    return data.to(device)


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    arr = (t * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def collect_inputs(path: Path):
    if path.is_file():
        return [path]

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="Run inference with UpsampleNN")
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="our_model.pth", help="Path to model weights")
    parser.add_argument("--upscale", type=int, default=2, help="Upscale factor used at training")
    parser.add_argument("--res-blocks", type=int, default=16, help="Number of residual blocks")
    parser.add_argument("--channels", type=int, default=32, help="Feature channels")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    model_path = Path(args.model)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    files = collect_inputs(input_path)
    if not files:
        raise RuntimeError(f"No supported image files found under: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = UpsampleNN(
        upscale_factor=args.upscale,
        num_res_blocks=args.res_blocks,
        channels=args.channels,
    ).to(device)
    load_state_dict_compatible(model, model_path, device)
    model.eval()

    with torch.no_grad():
        for src in files:
            img = Image.open(src).convert("RGB")
            inp = image_to_tensor(img, device)
            pred = model(inp)
            out_img = tensor_to_image(pred)

            out_name = f"{src.stem}_nnx{args.upscale}{src.suffix.lower()}"
            out_img.save(output_dir / out_name)
            print(f"Saved: {output_dir / out_name}")


if __name__ == "__main__":
    main()
