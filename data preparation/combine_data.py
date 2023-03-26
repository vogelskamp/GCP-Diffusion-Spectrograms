import os

import numpy as np
from tqdm import tqdm

FOLDER_PATH = r"/Users/samvogelskamp/Desktop/Uni/Audio Data Science/Diffusion-Spectrograms/data256"

for base, dirs, files in os.walk(FOLDER_PATH):

    print(f'Combining {len(files)} files in {FOLDER_PATH}')

    result = []
    for file in tqdm(files):

        with open(FOLDER_PATH + "/" + file, 'rb') as f:
            data = np.load(f)

            result.append(data)

    with open(f"data256.npy", 'wb') as f:
        # save to file
        np.save(f, np.array(result))
