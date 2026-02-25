import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter




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
    def __init__(
        self, upscale_factor: int = 2, num_res_blocks: int = 16, channels: int = 32
    ):
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

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def image_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
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


def has_option(args, option_name: str) -> bool:
    return any(tok == option_name for tok in args)


def build_ray_command(
    ray_bin: Path,
    scene: Path,
    output_image: Path,
    width: int,
    depth: int,
    render_json: Optional[str],
    render_cubemap: Optional[str],
    ray_args: List[str],
):
    cmd = [str(ray_bin)]

    if not has_option(ray_args, "-r"):
        cmd += ["-r", str(depth)]
    if not has_option(ray_args, "-w"):
        cmd += ["-w", str(width)]
    if render_json is not None and not has_option(ray_args, "-j"):
        cmd += ["-j", render_json]
    if render_cubemap is not None and not has_option(ray_args, "-c"):
        cmd += ["-c", render_cubemap]

    cmd += ray_args
    cmd += [str(scene), str(output_image)]
    return cmd


def run_ray_command(cmd: List[str]):
    print(f"[ray] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def np_rgb01(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-10) -> float:
    mse = np.mean((pred - target) ** 2)
    return 10.0 * np.log10(1.0 / (mse + epsilon))


def save_comparison_strip(
    ref_img: Image.Image,
    bicubic_img: Image.Image,
    gaussian_img: Image.Image,
    nn_img: Image.Image,
    output_path: Path,
):
    w, h = ref_img.size
    canvas = Image.new("RGB", (w * 4, h))
    canvas.paste(ref_img, (0, 0))
    canvas.paste(bicubic_img, (w, 0))
    canvas.paste(gaussian_img, (2 * w, 0))
    canvas.paste(nn_img, (3 * w, 0))
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="NN upsampling/antialiasing with optional ray-tracer render pass"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", help="Input image file or directory")
    source_group.add_argument("--scene", help="Scene file to render via ray tracer first")

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

    parser.add_argument("--reference", help="Optional ground-truth image for metric comparison")
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save a side-by-side strip (ref, bicubic, gaussian, nn) when reference is provided",
    )

    parser.add_argument("--ray-bin", default="build/bin/ray", help="Ray tracer binary path")
    parser.add_argument(
        "--render-width",
        type=int,
        default=320,
        help="Low-res width for the pre-NN ray-traced render",
    )
    parser.add_argument(
        "--render-depth", type=int, default=5, help="Recursion depth used for render mode"
    )
    parser.add_argument("--render-json", help="Optional ray-tracer json settings file")
    parser.add_argument("--render-cubemap", help="Optional cubemap file for render mode")
    parser.add_argument(
        "--render-reference",
        action="store_true",
        help="In scene mode, also render high-res reference at width*upscale for baseline comparison",
    )
    parser.add_argument(
        "--ray-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Pass-through args appended directly to ray tracer command",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    reference_path = Path(args.reference) if args.reference else None

    if args.scene:
        scene_path = Path(args.scene)
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene path does not exist: {scene_path}")

        ray_bin = Path(args.ray_bin)
        if not ray_bin.exists():
            raise FileNotFoundError(f"Ray tracer binary does not exist: {ray_bin}")

        render_dir = output_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)

        lr_render = render_dir / f"{scene_path.stem}_lr_w{args.render_width}.png"
        cmd = build_ray_command(
            ray_bin=ray_bin,
            scene=scene_path,
            output_image=lr_render,
            width=args.render_width,
            depth=args.render_depth,
            render_json=args.render_json,
            render_cubemap=args.render_cubemap,
            ray_args=args.ray_args,
        )
        run_ray_command(cmd)

        if args.render_reference:
            ref_width = args.render_width * args.upscale
            ref_render = render_dir / f"{scene_path.stem}_ref_w{ref_width}.png"
            ref_cmd = build_ray_command(
                ray_bin=ray_bin,
                scene=scene_path,
                output_image=ref_render,
                width=ref_width,
                depth=args.render_depth,
                render_json=args.render_json,
                render_cubemap=args.render_cubemap,
                ray_args=args.ray_args,
            )
            run_ray_command(ref_cmd)
            reference_path = ref_render

        input_path = lr_render
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = collect_inputs(input_path)
    if not files:
        raise RuntimeError(f"No supported image files found under: {input_path}")

    if reference_path is not None and len(files) != 1:
        raise RuntimeError("--reference currently supports single-image inference only")

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
            nn_img = tensor_to_image(pred)

            out_name = f"{src.stem}_nnx{args.upscale}{src.suffix.lower()}"
            nn_out_path = output_dir / out_name
            nn_img.save(nn_out_path)
            print(f"Saved NN output: {nn_out_path}")

            if reference_path is None:
                continue

            ref_img = Image.open(reference_path).convert("RGB")
            if ref_img.size != nn_img.size:
                ref_img = ref_img.resize(nn_img.size, Image.BICUBIC)

            bicubic_img = img.resize(nn_img.size, Image.BICUBIC)
            gaussian_img = bicubic_img.filter(ImageFilter.GaussianBlur(radius=1.0))

            bicubic_out = output_dir / f"{src.stem}_bicubicx{args.upscale}.png"
            gaussian_out = output_dir / f"{src.stem}_gaussianx{args.upscale}.png"
            bicubic_img.save(bicubic_out)
            gaussian_img.save(gaussian_out)
            print(f"Saved baseline output: {bicubic_out}")
            print(f"Saved baseline output: {gaussian_out}")

            ref_np = np_rgb01(ref_img)
            nn_np = np_rgb01(nn_img)
            bic_np = np_rgb01(bicubic_img)
            gau_np = np_rgb01(gaussian_img)

            psnr_nn = psnr(nn_np, ref_np)
            psnr_bic = psnr(bic_np, ref_np)
            psnr_gau = psnr(gau_np, ref_np)

            print("Comparison against reference:")
            print(f"  NN       PSNR: {psnr_nn:.3f} dB")
            print(f"  Bicubic  PSNR: {psnr_bic:.3f} dB")
            print(f"  Gaussian PSNR: {psnr_gau:.3f} dB")
            print(f"  NN - Bicubic:  {psnr_nn - psnr_bic:+.3f} dB")
            print(f"  NN - Gaussian: {psnr_nn - psnr_gau:+.3f} dB")

            if args.save_comparison:
                strip_path = output_dir / f"{src.stem}_comparison.png"
                save_comparison_strip(
                    ref_img=ref_img,
                    bicubic_img=bicubic_img,
                    gaussian_img=gaussian_img,
                    nn_img=nn_img,
                    output_path=strip_path,
                )
                print(f"Saved comparison strip: {strip_path}")


if __name__ == "__main__":
    main()
