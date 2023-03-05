from data_set import GCPSpectrogramSet, SpectrogramSet
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

import librosa.display
import matplotlib
import torch

import json

from google.cloud import storage


# main thread shenanigans
matplotlib.use('agg')


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_image(sample, image_height, image_width, epoch, loop_idx, bucket_name):
    # convert back to db
    for idx in range(0, len(sample)):
        sample[idx] = [x * 80 - 80 for x in sample[idx]]

        M_db = np.array(sample)

        fig = plt.figure()
        fig.set_size_inches(image_width / 100,
                            image_height / 100, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        librosa.display.specshow(M_db, ax=ax, cmap='Greys_r')

        plt.savefig(f"/gcs/{bucket_name}/result_{epoch}_{loop_idx}.png",
                    bbox_inches='tight', pad_inches=0)
        plt.close()


def get_data(args):

    dataset = GCPSpectrogramSet(
        file_name=args.dataset, bucket_name=args.bucket_name)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader
