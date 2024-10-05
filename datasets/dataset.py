import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class Dataset(Dataset):
    def __init__(self, path, device='cuda', transform=None):
        self.data = torch.tensor(np.array(h5py.File(path, 'r')['data']))
        self.transform = transform
        self.device = device
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx])
        return self.data[idx]

def get_dataloader(path, batch_size, device='cuda', transform=None):
    dataset = Dataset(path, device, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)