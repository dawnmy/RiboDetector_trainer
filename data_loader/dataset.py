import random
from itertools import chain, cycle
from torch.utils.data import Dataset, IterableDataset


class SeqFeature(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


#! TODO
class SeqDataset(IterableDataset):

    def __init__(self, data_list, batch_size, transform=None):
        self.data_list = data_list
        self.batch_size = batch_size
        self.transform = transform

    @property
    def shuffle_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        for

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
