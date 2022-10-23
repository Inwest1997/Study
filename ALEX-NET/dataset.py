from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os


class FashionDataset(Dataset):
    def __init__(self, transform = None, name = 'fashion-mnist_train'):
        self.transform = transform
        self.path = os.path.join('Dataset', name)
        self.data = list(pd.read_csv(f'{self.path}.csv').values)
        self.img = np.asarray([i[1:] for i in self.data]).reshape(-1, 28, 28, 1).astype('float32')
        self.label = np.asarray([i[0] for i in self.data])

    def __getitem__(self, index) :
        label = self.label[index]
        image = self.img[index]

        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.img)