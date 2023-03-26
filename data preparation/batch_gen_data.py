import math
import os
import tkinter.filedialog

import audio_utils as utils
import librosa
import numpy as np
import soundfile as sf
from augment import augment
from tqdm import tqdm

MAX_LENGTH = 12
TIME_STEPS = 256
N_MELS = 64


# ORIGIN_FOLDER_PATH = tkinter.filedialog.askdirectory(
#     title='Select Origin Folder')

ORIGIN_FOLDER_PATH = r"/Users/samvogelskamp/Desktop/Uni/Audio Data Science/Diffusion-Spectrograms/audio"

# DESTINATION_FOLDER_PATH = tkinter.filedialog.askdirectory(
#     title='Select Destination Folder') + "/"

DESTINATION_FOLDER_PATH = r"/Users/samvogelskamp/Desktop/Uni/Audio Data Science/Diffusion-Spectrograms/data256"

last_idx = int(len(os.listdir(DESTINATION_FOLDER_PATH)) / 21)

for base, dirs, files in os.walk(ORIGIN_FOLDER_PATH):

    files = files[last_idx:]

    print(f'Augmenting {base} starting at index {last_idx}')

    for file in tqdm(files):
        # Append Filepath to current Filepath
        currentFilepath = ORIGIN_FOLDER_PATH + "/" + file

        # load File
        data, sr = librosa.load(currentFilepath, sr=None)

        TARGET_SAMPLES = sr * MAX_LENGTH

        if len(data) > TARGET_SAMPLES:
            print(file, " is too long, skipping...")
            continue

        # expand data through augmentation
        expanded_data = augment(data, sr, file.split('.')[-2])

        # generate mel spectrograms
        for audio, name in expanded_data:

            # add padding and fadeout
            utils.apply_fadeout(audio, sr)

            audio = np.append(audio, np.zeros(
                TARGET_SAMPLES - len(audio), dtype=np.float32))

            hop_length = math.floor(len(audio)/TIME_STEPS)

            start_sample = 0
            length_samples = TIME_STEPS * hop_length
            window = audio[start_sample:start_sample+length_samples - 1]

            M = librosa.feature.melspectrogram(
                y=window, sr=sr, n_mels=N_MELS, hop_length=hop_length)

            # convert power to db
            M_db = librosa.power_to_db(M, ref=np.max)

            # convert 0 to -80 -> 1 to 0
            for iy, ix in np.ndindex(M_db.shape):
                M_db[iy, ix] = 1 - M_db[iy, ix] / -80

            # convert to 3D array
            M_db = M_db[np.newaxis, :, :]

            # save
            with open(f"{DESTINATION_FOLDER_PATH}/{name}.npy", 'wb') as f:
                # save to file
                np.save(f, M_db)
