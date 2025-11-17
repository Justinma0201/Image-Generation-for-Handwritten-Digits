#!/usr/bin/env python3
# coding: utf-8

import os
from glob import glob
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import wandb
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, folder_path):
        self.image_path = sorted(glob(os.path.join(folder_path, "*.png")))
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (C, H, W), [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -> [-1,1]
        ])
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, index):
        img_path = self.image_path[index]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.)
        return emb  # (B, dim)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim, num_group=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_group, out_channel), out_channel)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_group, out_channel), out_channel)
        self.residual_conv = nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)
        h = self.conv1(x)
        h = self.norm1(h)
        time = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time
        h = self.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return h + residual

class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.init_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)
        self.down_block1 = ResidualBlock(base_channels, base_channels, time_emb_dim, num_group=8)
        self.down_sample1 = nn.MaxPool2d(2)  # 28 -> 14
        self.down_block2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim, num_group=8)
        self.down_sample2 = nn.MaxPool2d(2)  # 14 -> 7
        self.mid_block1 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, num_group=8)
        self.mid_block2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, num_group=8)
        self.up_sample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.up_block1 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim, num_group=8)
        self.up_sample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.up_block2 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim, num_group=8)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        x = self.init_conv(x)
        skip1 = self.down_block1(x, t_emb)
        x = self.down_sample1(skip1)
        skip2 = self.down_block2(x, t_emb)
        x = self.down_sample2(skip2)
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)
        x = self.up_sample1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_block1(x, t_emb)
        x = self.up_sample2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_block2(x, t_emb)
        predicted_noise = self.final_conv(x)
        return predicted_noise

def _extract(coefficients, t, x_shape):
    batch_size = t.shape[0]
    out = coefficients[t].to(coefficients.device)
    return out.reshape(batch_size, 1, 1, 1)

class DiffusionModel:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def compute_loss(self, model, x_0):
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        t_tensor = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        predicted_noise = model(x_t, t_tensor)
        betas_t = _extract(self.betas, t_tensor, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t_tensor, x_t.shape)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        if t == 0:
            return model_mean
        else:
            posterior_variance_t = _extract(self.posterior_variance, t_tensor, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, batch_size):
        model.eval()
        x_t = torch.randn((batch_size, 3, 28, 28), device=self.device)
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(model, x_t, t)
        x_0 = (x_t + 1.) / 2.
        x_0 = x_0.clamp(0., 1.)
        return x_0

def run_training(config):
    run = wandb.init(
        project="ddpm-mnist-hw3",
        name=f"epoch_{config.epochs}_lr_{config.lr}_bs_{config.batch_size}",
        config=vars(config)
    )
    os.makedirs(config.save_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading dataset from: {config.data_path}")
    dataset = MNISTDataset(config.data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = UNet().to(device)
    diffusion = DiffusionModel(timesteps=config.timesteps, device=device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    wandb.watch(model, log="all", log_freq=100)
    print("Start training...")
    global_step = 0

    for epoch in range(config.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        model.train()
        for i, images in enumerate(progress_bar):
            images = images.to(device)
            loss = diffusion.compute_loss(model, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()}, step=global_step)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.epochs:
            save_file = os.path.join(config.save_path, f"unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_file)
            print(f"Epoch {epoch+1} finished, model saved to {save_file}")
            print("Generating sample images for wandb...")
            model.eval()
            with torch.no_grad():
                sample_images = diffusion.sample(model, batch_size=8)  # float [0,1]
            wandb_images = []
            for img in sample_images:
                np_img = (img * 255.).permute(1, 2, 0).cpu().numpy().astype('uint8')
                wandb_images.append(wandb.Image(np_img))
            wandb.log({"generated_examples": wandb_images}, step=global_step)
            print("Sample images logged to wandb.")
        else:
            print(f"Epoch {epoch+1} finished.")
    print("Training finished")
    run.finish()

def run_sampling(args):
    os.makedirs(args.output_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Please check --ckpt_path")
        return
    model.eval()
    diffusion = DiffusionModel(timesteps=args.timesteps, device=device)
    print(f"Start generating {args.num_images} images...")
    count = 1
    with torch.no_grad():
        while count <= args.num_images:
            current_batch_size = min(args.batch_size, args.num_images - count + 1)
            if current_batch_size <= 0:
                break
            generated_images = diffusion.sample(model, current_batch_size)  # float [0,1]
            for i in range(current_batch_size):
                if count > args.num_images:
                    break
                img_tensor = generated_images[i]
                np_img = (img_tensor * 255.).permute(1, 2, 0).cpu().numpy().astype('uint8')
                filename = f"{count:05d}.png"
                filepath = os.path.join(args.output_folder, filename)
                Image.fromarray(np_img).save(filepath)
                count += 1
            print(f"Generated {count-1}/{args.num_images} images...")
    print(f"Successfully generated {args.num_images} images to {args.output_folder}")

@torch.no_grad()
def run_visualization(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load(config.ckpt_path, map_location=device))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Please check --ckpt_path: {config.ckpt_path}")
        return
    model.eval()
    
    diffusion = DiffusionModel(timesteps=config.timesteps, device=device)

    T = config.timesteps
    record_steps = torch.linspace(T - 1, 0, 8).long().tolist() 
    print(f"Recording snapshots at t = {record_steps}")

    batch_size = 8
    x_t = torch.randn((batch_size, 3, 28, 28), device=device)
    images_list = []
    
    for t in reversed(range(0, T)):
        if t in record_steps and t != 0:
            print(f"Recording snapshot at t={t}")
            x_t_denoised = (x_t + 1.) / 2.
            x_t_denoised = x_t_denoised.clamp(0., 1.)
            images_list.append(x_t_denoised)
        
        x_t = diffusion.p_sample(model, x_t, t)
    
    if 0 in record_steps:
        print("Recording snapshot at t=0")
        x_0_denoised = (x_t + 1.) / 2.
        x_0_denoised = x_0_denoised.clamp(0., 1.)
        images_list.append(x_0_denoised)

    final_grid_tensors = []
    recorded_t = [step for step in record_steps if step != 0]
    
    for tensor in images_list[:-1]:
        final_grid_tensors.append(tensor)
    
    final_grid_tensors.append(images_list[-1])
    
    if len(final_grid_tensors) != 8:
        print(f"Warning: Expected 8 snapshots, but got {len(final_grid_tensors)}. Check linspace logic.")
        
    grid_tensor = torch.cat(final_grid_tensors, dim=0)

    grid_image = torchvision.utils.make_grid(grid_tensor, nrow=8)
    
    output_filename = "diffusion_process_8x8.png"
    torchvision.utils.save_image(grid_image, output_filename)
    
    print(f"Success! 8x8 visualization grid saved to {output_filename}")

if __name__ == "__main__":
    
    MODE = "train" 

    # config class
    class Config:
        pass
    config = Config()
    
    config.timesteps = 1000

    config.data_path = "./data"
    config.save_path = "./checkpoints"
    config.epochs = 100
    config.batch_size = 128
    config.lr = 1e-4
    config.save_every = 10

    config.output_folder = "./image_114064558"
    config.num_images = 10000
    
    config.ckpt_path = "./checkpoints/unet_epoch_100.pth"
    
    if MODE == "train":
        run_training(config)
        
    elif MODE == "sample":
        config.batch_size = 64 
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--ckpt_path", type=str, default=config.ckpt_path)
        parser.add_argument("--output_folder", type=str, default=config.output_folder)
        parser.add_argument("--num_images", type=int, default=config.num_images)
        parser.add_argument("--batch_size", type=int, default=config.batch_size)
        parser.add_argument("--timesteps", type=int, default=config.timesteps)
        
        args = parser.parse_args()
        
        args.data_path = config.data_path
        args.save_path = config.save_path
        args.epochs = config.epochs
        args.lr = config.lr
        args.save_every = config.save_every
        
        run_sampling(args)
        
    elif MODE == "visualize":
        run_visualization(config)
        
    else:
        print(f"錯誤：未知的 MODE '{MODE}'。請設定為 'train', 'sample' 或 'visualize'。")