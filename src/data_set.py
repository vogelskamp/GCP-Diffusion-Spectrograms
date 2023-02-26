import numpy as np
from torch.utils.data import Dataset


class SpectrogramSet(Dataset):

    def __init__(self, data_path, transform=None):

        self.data_path = data_path
        self.data = self.load_data(self.data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        target = self.data[idx]

        if self.transform:
            target = self.transform(target)

        return target

    def load_data(self, path):

        with open(path, 'rb') as f:
            data = np.load(f)
            # data = data[0:10, :, :, :]
            print(f"Dataset shape: {data.shape}")
            return data

if __name__ == '__main__':
    test = SpectrogramSet(
        data_path="C:/Users/student-isave/Documents/Diffusion-Spectrograms/audio_data")
