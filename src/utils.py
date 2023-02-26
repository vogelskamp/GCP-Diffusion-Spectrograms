import os

import librosa.display
import matplotlib
import torch

# main thread shenanigans
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from data_set import SpectrogramSet


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_batch_images(batch, save_path, image_height, image_width):

    results = []
    for sample_idx in range(0, len(batch)):
        results.append([])

        sample = np.squeeze(batch[sample_idx], axis=0)
        # convert back to db
        for idx in range(0, len(sample)):
            results[sample_idx].append([x * 80 - 80 for x in sample[idx]])            

    for idx in range(0, len(results)):
        M_db = np.array(results[idx])

        fig = plt.figure()
        fig.set_size_inches(image_width / 100, image_height / 100, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        librosa.display.specshow(M_db, ax=ax, cmap='Greys_r')

        if not os.path.isdir(save_path): os.mkdir(save_path)

        plt.savefig(f"{save_path}/result_{idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

def save_image(image, path, image_height, image_width, loop_idx, **kwargs):
    # result = []
    sample = np.squeeze(image, axis=0)
    # convert back to db
    for idx in range(0, len(sample)):
        sample[idx] = [x * 80 - 80 for x in sample[idx]]

        M_db = np.array(sample)

        fig = plt.figure()
        fig.set_size_inches(image_width / 100, image_height / 100, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        librosa.display.specshow(M_db, ax=ax, cmap='Greys_r')

        if not os.path.isdir(path): os.mkdir(path)

        plt.savefig(f"{path}/result_{loop_idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
def save_images(images, path, image_height, image_width, **kwargs):
    save_batch_images(images.to('cpu').numpy(), path, image_height, image_width)
    # ndarr = (ndarr * 255).astype(np.uint8)
    # im = Image.fromarray(ndarr)
    # im.save(path)


def get_data(args):
    # transforms = torchvision.transforms.Compose([
    #     # args.image_size + 1/4 *args.image_size
    #     # torchvision.transforms.Resize(
    #     #     args.image_size[0] + 1/4 * args.image_size[0]),
    #     # torchvision.transforms.RandomResizedCrop(
    #     #     args.image_size, scale=(0.8, 1.0)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                             torchvision.transforms.Normalize(
    #                                                 (0.5,), (0.5,))
    #                                              ])

    dataset = SpectrogramSet(data_path=args.dataset_path)
    # dataset = SpectrogramSet(data_path=args.dataset_path, transform=transforms)
    # dataset = torchvision.datasets.FakeData(
    #     size=5, image_size=(1, args.image_size[0], args.image_size[1]), transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
