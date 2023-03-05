import os
import random
import shutil

from tqdm import tqdm

basePath = r"S:\Code\_Uni\Diffusion-Spectrograms\gcp_data\images"

dict = {}
print("Creating dict")
for file in tqdm(os.listdir(basePath)):

    epoch = file.split("_")[1]

    if epoch in dict:
        dict[epoch].append(file)
    else:
        dict[epoch] = [file]

print("Copying images")
for epoch, files in tqdm(dict.items()):

    targetIdx = random.randint(0, len(files) - 1)

    shutil.copy(f"{basePath}/{files[targetIdx]}", f"S:\Code\_Uni\Diffusion-Spectrograms\data_analysis\gif2\{epoch}.png")

    