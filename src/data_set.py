import numpy as np
from google.cloud import storage
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


class GCPSpectrogramSet(Dataset):
    def __init__(self, file_name, bucket_name, transform=None):

        self.file_name = file_name
        self.bucket_name = bucket_name
        self.data = self.load_data(self.file_name, self.bucket_name)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        target = self.data[idx]

        if self.transform:
            target = self.transform(target)

        return target

    def load_data(self, file_name, bucket_name):
        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        # file = open(path, 'r')
        blob = bucket.blob(file_name)

        print(
            f"Fetching dataset {file_name} from GCP bucket {bucket_name}")

        blob.download_to_filename('temp_file_name')

        with open('temp_file_name', 'rb') as file:
            data = np.load(file)

            print(f"Dataset shape: {data.shape}")
            return data


if __name__ == '__main__':
    test = SpectrogramSet(
        data_path="C:/Users/student-isave/Documents/Diffusion-Spectrograms/audio_data")
