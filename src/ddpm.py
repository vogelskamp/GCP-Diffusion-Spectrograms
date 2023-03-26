import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from modules import UNet
from utils import *


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
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():

            # generate random noise
            x = torch.randn(
                (n, self.c_in, self.img_size[0], self.img_size[1])).to(self.device)
            
            # loop over time steps
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                
                # predict noise
                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # subtract noise
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                             * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x


def train(args):

    print("Fetching data")
    dataloader = get_data(args)

    model = UNet(
        device=args.device, img_height=args.image_size[0], img_width=args.image_size[1])

    model = nn.DataParallel(model)

    model.to(args.device)

    print(f"Using {torch.cuda.device_count()} GPUs")
    print(
        f"Created model with {sum([p.numel() for p in model.parameters()])} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=args.device)

    # training loop
    for epoch in range(0, args.epochs):
        print(f"Starting epoch {epoch}:")

        pbar = tqdm(dataloader)
        for images in pbar:

            # load the image to the hardware device
            images = images.to(args.device)

            # sample time steps
            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)

            # noise images and save the real noise as label
            x_t, noise = diffusion.noise_images(images, t)

            # predict noise using the U-Net model
            predicted_noise = model(x_t, t)

            # calculate loss based on the meansquare error between label and prediction
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        # sample new result images
        sampled_image = diffusion.sample(model, n=images.shape[0])
        samples = sampled_image.to('cpu').numpy()
        for idx, data in enumerate(samples):
            save_image(np.squeeze(data, axis=0),
                       args.image_size[0], args.image_size[1], epoch, idx, args.result_bucket)

        # save the current state of the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'/gcs/{args.result_bucket}/ckpt.pt')


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=100)
    parser.add_argument("-bs", "--batch_size", default=5)
    parser.add_argument("-is", "--image_size", default=(64, 256))
    parser.add_argument("-lr", "--learning_rate", default=3e-4)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-ds", "--dataset",
                        default='data256_test.npy')
    parser.add_argument("-bn", "--bucket_name",
                        default='diffusion-project-data')
    parser.add_argument("-rb", "--result_bucket",
                        default='diffusion-project-results-na')
    parser.add_argument("-n", "--name", default="DDPM_Unconditional256x")
    args = parser.parse_args()

    print(f"Received args: {args}")

    train(args)


if __name__ == '__main__':
    launch()
