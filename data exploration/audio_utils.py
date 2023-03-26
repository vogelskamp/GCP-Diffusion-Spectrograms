import math

import librosa
import matplotlib.image as mpimg
import noisereduce as nr
import numpy as np
from PIL import Image


def apply_fadeout(audio, sr, duration=0.01):
    # convert to audio indices (samples)
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length

    # compute fade out curve
    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)

    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve
    
def load_and_convert_to_db(file, sr=None, n_mels=128, time_steps=512):
    data, sr = librosa.load(file, sr=None)

    hop_length = math.floor(len(data)/time_steps)
    start_sample = 0
    length_samples = time_steps * hop_length
    window = data[start_sample:start_sample+length_samples - 1]

    M = librosa.feature.melspectrogram(
        y=window, sr=sr, n_mels=n_mels, hop_length=hop_length)

    # convert power to db
    return librosa.power_to_db(M, ref=np.max)

def image_to_audio(path, hop_length, sr=44100):
    with Image.open(path) as img:

        M_db = []
        for y in range(img.height):
            M_db.append([])
            for x in range(img.width):
                value = img.getpixel((x,y))[0]
                M_db[y].append((value / 255) * 80 - 80)

        M_db = np.array(M_db)
        M_db = np.flip(M_db, axis=0)

        M = librosa.db_to_power(M_db)
        audio = librosa.feature.inverse.mel_to_audio(M, sr=sr, hop_length=hop_length)
        return nr.reduce_noise(audio, sr=sr)
