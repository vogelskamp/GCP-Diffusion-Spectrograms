import cv2
import numpy as np

with open(r"S:\Code\_Uni\Diffusion-Spectrograms\data\13_01_2023_14_11_25.npy", 'rb') as f:
    arr = np.load(f)

    avgs = arr.mean(axis=(0,1)) * 255
    avgs = np.flip(avgs, axis=0)

    cv2.imwrite("./test.png", avgs)