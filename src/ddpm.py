import logging
import os

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from modules import UNet
from utils import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, img_size=(128, 128), c_in=1, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.c_in = c_in
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn(
                (n, self.c_in, self.img_size[0], self.img_size[1])).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                             * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.name)
    device = args.device
    dataloader = get_data(args)

    start_epoch = 0
    model = UNet(device=device, img_height=args.image_size[0], img_width=args.image_size[1]).to(device)
    if os.path.isfile(os.path.join("models", args.name, f"ckpt.pt")):
        checkpoint = torch.load(os.path.join("models", args.name, f"ckpt.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Found checkpoint at epoch {start_epoch}, loading model")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    len_dataloader = len(dataloader)

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for loop_idx, images in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        logging.info(f"Sampling {5} new images....")
        for loop_idx in tqdm(range(0, 5)):
            sampled_image = diffusion.sample(model, n=images.shape[0])
            save_image(np.squeeze(sampled_image.to('cpu').numpy(), axis=0), os.path.join(
                "results", args.name, str(epoch)), args.image_size[0], args.image_size[1], loop_idx)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(
            "models", args.name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=200)
    parser.add_argument("-bs", "--batch-size", default=1)
    parser.add_argument("-is", "--image-size", default=(128, 128))
    parser.add_argument("-lr", "--learning-rate", default=3e-4)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-ds", "--dataset", default="C:/Users/student-isave/Documents/Diffusion-Spectrograms/data/13_01_2023_14_11_25.npy")
    parser.add_argument("-n", "--name", default="DDPM_Unconditional128x")
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
