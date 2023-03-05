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
    device = args.device

    print("Fetching data")
    dataloader = get_data(args)

    start_epoch = 0

    model = UNet(
        device=device, img_height=args.image_size[0], img_width=args.image_size[1])

    model = nn.DataParallel(model)

    model.to(device)

    print(f"Using {torch.cuda.device_count()} GPUs")

    print(
        f"Created model with {sum([p.numel() for p in model.parameters()])} parameters")

    # if os.path.isfile(os.path.join("models", args.name, f"ckpt.pt")):
    #     checkpoint = torch.load(os.path.join("models", args.name, f"ckpt.pt"))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"Found checkpoint at epoch {start_epoch}, loading model")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)

    for epoch in range(start_epoch, args.epochs):
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for images in pbar:
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        sampled_image = diffusion.sample(model, n=images.shape[0])

        samples = sampled_image.to('cpu').numpy()

        for idx, data in enumerate(samples):
            save_image(np.squeeze(data, axis=0),
                       args.image_size[0], args.image_size[1], epoch, idx, args.result_bucket)

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
