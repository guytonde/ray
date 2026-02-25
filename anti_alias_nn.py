import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# 1. The Model (SRCNN)
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

# 2. Rigorous Training with Patches
def train_model(image_paths, epochs=200):
    print(f"--- Training Phase: Learning Edge Geometry ---")
    model = SRCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # We extract small 32x32 patches to give the network more samples to learn from
    patch_size = 32
    scale = 2
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            # Create a blurry/low-res version
            w, h = img.size
            lr = img.resize((w // scale, h // scale), Image.BICUBIC).resize((w, h), Image.BICUBIC)
            
            # Convert to numpy to extract patches
            hr_np = np.array(img) / 255.0
            lr_np = np.array(lr) / 255.0
            
            # Extract 5 random patches per image per epoch
            for _ in range(5):
                iy = np.random.randint(0, h - patch_size)
                ix = np.random.randint(0, w - patch_size)
                
                hr_patch = hr_np[iy:iy+patch_size, ix:ix+patch_size].transpose(2,0,1)
                lr_patch = lr_np[iy:iy+patch_size, ix:ix+patch_size].transpose(2,0,1)
                
                hr_t = torch.FloatTensor(hr_patch).unsqueeze(0)
                lr_t = torch.FloatTensor(lr_patch).unsqueeze(0)

                optimizer.zero_grad()
                output = model(lr_t)
                loss = criterion(output, hr_t)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/ (len(image_paths)*5):.6f}")
            
    return model

# 3. PSNR Calculation (Rigorous Metric)
def calculate_psnr(img1, img2):
    # Higher is better. Classical usually gets 25-28, NN should get 30+
    mse = np.mean((np.array(img1).astype(float) - np.array(img2).astype(float)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# 4. Processing and Comparison
def process_comparison(model, img_path, output_folder):
    filename = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    scale = 2

    # A. Classical AA (Gaussian)
    # This is what a standard renderer does: supersample then blur
    classical_aa = img.resize((w * scale, h * scale), Image.BICUBIC)
    classical_aa = classical_aa.filter(ImageFilter.GaussianBlur(radius=1.2))

    # B. Neural Network
    nn_input = img.resize((w * scale, h * scale), Image.BICUBIC)
    input_t = transforms.ToTensor()(nn_input).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output_t = model(input_t).clamp(0, 1)
        nn_img = transforms.ToPILImage()(output_t.squeeze(0))

    # C. Visual Comparison with ZOOM
    # We zoom into a 100x100 area to see the actual antialiasing quality
    zoom_box = (w//scale, h//scale, w//scale + 100, h//scale + 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(classical_aa)
    axes[0, 0].set_title("Classical AA (Gaussian Blur)")
    
    axes[0, 1].imshow(nn_img)
    axes[0, 1].set_title("Neural Network (SRCNN)")
    
    # Zoomed Views
    axes[1, 0].imshow(classical_aa.crop(zoom_box))
    axes[1, 0].set_title("Classical Zoom (Blurry)")
    
    axes[1, 1].imshow(nn_img.crop(zoom_box))
    axes[1, 1].set_title("NN Zoom (Sharp Edges)")

    for ax in axes.flatten(): ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"comp_{filename}"))
    plt.close()
    
    print(f"Result saved for {filename}")

def main():
    input_dir = "nn_inputs"
    output_dir = "nn_upscaled_results"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))

    if not image_paths:
        print("Put PNGs in 'nn_inputs' folder.")
        return

    # 1. Train once on the whole dataset
    trained_model = train_model(image_paths, epochs=200)

    # 2. Compare results
    for path in image_paths:
        process_comparison(trained_model, path, output_dir)

if __name__ == "__main__":
    main()