import os
from glob import glob
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.io as io
from tqdm import tqdm
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def safe_psnr(img1, img2, data_range=1.0, epsilon=1e-10):
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10((data_range ** 2) / (mse + epsilon))




# ---------------------------
# Dataset (random crops + flips for training, full image for eval)
# ---------------------------
class ImagePairDataset(Dataset):
    def __init__(self, folder, upscale_factor=2, patch_size=128, augment=True, patches_per_image=16):
        self.files = sorted(glob(os.path.join(folder, "*.png")))
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.augment = augment
        self.patches_per_image = patches_per_image
        self._train_mode = True


    def eval_mode(self):
        self._train_mode = False


    def train_mode(self):
        self._train_mode = True


    def __len__(self):
        if self._train_mode:
            return len(self.files) * self.patches_per_image
        return len(self.files)


    def __getitem__(self, idx):
        # Cycle through the images
        file_idx = idx % len(self.files)
        hr = io.read_image(self.files[file_idx]).float() / 255.0  # Load as tensor [C,H,W]
        fname = os.path.basename(self.files[file_idx])


        if self._train_mode:
            # Random crop (same behavior as PIL crop)
            h, w = hr.shape[1], hr.shape[2]
            if w >= self.patch_size and h >= self.patch_size:
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)
                hr = hr[:, y:y + self.patch_size, x:x + self.patch_size]


            # Random flips (horizontal and vertical, mimicking PIL)
            if self.augment:
                if random.random() > 0.5:
                    hr = TF.hflip(hr)
                if random.random() > 0.5:
                    hr = TF.vflip(hr)


        # Bicubic resize (mimicking PIL's Image.BICUBIC)
        h, w = hr.shape[1], hr.shape[2]
        lr = TF.resize(hr, size=(h // self.upscale_factor, w // self.upscale_factor), interpolation=TF.InterpolationMode.BICUBIC)


        return lr, hr, fname



class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention from RCAN (Zhang et al., ECCV 2018)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        scale = self.fc(self.avg_pool(x))
        return x * scale


class ResidualBlock(nn.Module):
    """
    EDSR-style residual block (no BN) + RCAN channel attention.
    res_scale=0.1 stabilizes training of deep networks (Lim et al., CVPR 2017).
    """
    def __init__(self, channels=32, res_scale=0.1):
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
    """
    Key improvements over baseline:
      1. Global skip: bicubic upsample LR added to output so the net only
         learns the high-frequency residual (EDSR / VDSR principle).
      2. Long skip connection over all residual blocks (EDSR).
      3. Channel attention in every residual block (RCAN).
      4. Residual scaling (0.1) for training stability (EDSR).
      5. No BN -- consistent with EDSR finding that BN hurts SR quality.
    """
    def __init__(self, upscale_factor=2, num_res_blocks=16, channels=32):
        super().__init__()
        self.upscale_factor = upscale_factor


        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1)


        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, res_scale=0.1) for _ in range(num_res_blocks)]
        )


        self.conv_post_res = nn.Conv2d(channels, channels, kernel_size=3, padding=1)


        self.conv_pre_upsample = nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)


        self.conv_output = nn.Conv2d(channels, 3, kernel_size=3, padding=1)


    def forward(self, x):
        bicubic_up = F.interpolate(
            x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False
        )


        feat = self.conv_input(x)


        res = self.res_blocks(feat)
        res = self.conv_post_res(res)
        feat = feat + res


        feat = self.conv_pre_upsample(feat)
        feat = self.pixel_shuffle(feat)


        residual_out = self.conv_output(feat)
        return bicubic_up + residual_out



def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=100, save_path="our_model.pth"):
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


    for epoch in range(epochs):
        # Training phase
        loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        running_loss = 0.0
        for lr_imgs, hr_imgs, _ in loop:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                sr = model(lr_imgs)
                loss = criterion(sr, hr_imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n + 1), lr=f"{scheduler.get_last_lr()[0]:.2e}")


        scheduler.step()
       
        # Validation phase
        model.eval()  # Switch to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs, _ in val_dataloader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr = model(lr_imgs)
                loss = criterion(sr, hr_imgs)
                val_loss += loss.item()


        # Print loss after validation
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {running_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss / len(val_dataloader):.4f}")
       
        # Save the model after each epoch
        torch.save(model.state_dict(), save_path)


    print("Training finished.")
    print(f"Model saved as {save_path}")


def evaluate_and_save_results_fast(model, dataset, device, output_folder="comparison_results", batch_size=4):
    os.makedirs(output_folder, exist_ok=True)
    model.eval()
    dataset.eval_mode()


    # Wrap dataset in DataLoader for batching + parallel loading
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    psnr_nn, ssim_nn = [], []
    psnr_bic, ssim_bic = [], []
    psnr_gauss, ssim_gauss = [], []


    with torch.no_grad():
        for batch in loader:
            lr_batch, hr_batch, fnames = batch
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            sr_batch = model(lr_batch)


            for i in range(lr_batch.size(0)):
                hr = hr_batch[i]
                sr = sr_batch[i]
                lr = lr_batch[i]
                fname = fnames[i]


                # Convert to numpy on GPU (avoid CPU copies until needed)
                hr_np = hr.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
                sr_np = sr.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
               
                # Bicubic + Gaussian (CPU because PIL works on CPU)
                w, h = hr_np.shape[1], hr_np.shape[0]
                lr_img = transforms.ToPILImage()(lr.cpu())
                bic_img = lr_img.resize((w, h), Image.BICUBIC)
                bic_np = np.array(bic_img).astype(np.float32) / 255.0
                gauss_img = bic_img.filter(ImageFilter.GaussianBlur(radius=1))
                gauss_np = np.array(gauss_img).astype(np.float32) / 255.0


                # Metrics
                psnr_nn.append(safe_psnr(hr_np, sr_np, data_range=1.0))
                ssim_nn.append(ssim(hr_np, sr_np, channel_axis=2, data_range=1.0))
                psnr_bic.append(safe_psnr(hr_np, bic_np, data_range=1.0))
                ssim_bic.append(ssim(hr_np, bic_np, channel_axis=2, data_range=1.0))
                psnr_gauss.append(safe_psnr(hr_np, gauss_np, data_range=1.0))
                ssim_gauss.append(ssim(hr_np, gauss_np, channel_axis=2, data_range=1.0))


                # Side-by-side plots
                fig, axes = plt.subplots(1, 4, figsize=(16, 5))
                axes[0].imshow(hr_np)
                axes[0].set_title("HR Ground Truth")
                axes[1].imshow(bic_np)
                axes[1].set_title("Bicubic")
                axes[2].imshow(gauss_np)
                axes[2].set_title("Gaussian")
                axes[3].imshow(np.clip(sr_np, 0, 1))
                axes[3].set_title("Neural Net")
                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, fname))
                plt.close()


    print("Neural Net       PSNR: {:.2f}, SSIM: {:.4f}".format(np.mean(psnr_nn), np.mean(ssim_nn)))
    print("Bicubic Baseline PSNR: {:.2f}, SSIM: {:.4f}".format(np.mean(psnr_bic), np.mean(ssim_bic)))
    print("Gaussian Baseline PSNR: {:.2f}, SSIM: {:.4f}".format(np.mean(psnr_gauss), np.mean(ssim_gauss)))
    print(f"Side-by-side images saved in '{output_folder}'")







if __name__ == "__main__":
    train_folder = "DIV2K/DIV2K_train_HR"
    val_folder   = "DIV2K/DIV2K_valid_HR"
    upscale_factor = 2
    batch_size = 32
    epochs = 15
    learning_rate = 1e-4
    model_path = "our_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ImagePairDataset(
        train_folder,
        upscale_factor=upscale_factor,
        patch_size=128,
        augment=True,
        patches_per_image=16
    )
    val_dataset = ImagePairDataset(
        val_folder, upscale_factor=upscale_factor, augment=False, patches_per_image=1
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True)


    model = UpsampleNN(upscale_factor=upscale_factor, num_res_blocks=16, channels=32).to(device)


    if os.path.exists(model_path):
        print(f"Found existing model '{model_path}', skipping training...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss()
        print("Starting training...")
        train(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=epochs, save_path=model_path)


    # Evaluate on training and validation sets
    print("\nEvaluating on training domain (nn_inputs)...")
    train_domain_dataset = ImagePairDataset(
        "nn_inputs", upscale_factor=upscale_factor, augment=False, patches_per_image=1
    )
    evaluate_and_save_results_fast(model, train_domain_dataset, device, output_folder="comparison_results", batch_size=4)


    print("\nEvaluating on DIV2K validation set...")
    evaluate_and_save_results_fast(model, val_dataset, device, output_folder="comparison_results_val", batch_size=1)





