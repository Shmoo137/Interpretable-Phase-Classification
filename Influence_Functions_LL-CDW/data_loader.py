import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as F
import numpy as np

# generate dataset (it transforms numpy arrays to torch tensors)
class NumpyToPyTorch_DataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y, transform=None):
        self.X = torch.from_numpy(X).float()    # image
        self.Y = torch.from_numpy(Y).long()     # label for classification
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        label = self.Y[index]
        img = self.X[index]

        if self.transform:
            img = self.transform(img)

        return img, label

# it's super ugly, but it looks like I cannot pass additional argument to Dataset, so screw it
class SingleNumpyToPyTorch_DataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y, transform=None):
        """
        Args:
            path
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = torch.FloatTensor([X])    # image
        self.Y = torch.LongTensor([Y])  # single label

        # i, j = self.Y.size()[0], self.Y.size()[1]
        # self.Y = self.Y.view(i, 1, j)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        label = self.Y[index]
        img = self.X[index]

        if self.transform:
            img = self.transform(img)

        return img, label

class Downloader(object):
    def __init__(self, data_name, batch):
        self.data_name = data_name
        self.batch = batch

        if self.data_name is 'CDW_ED_size12_full':
            self.training_path = 'datasets/ED_size12_full_training.csv'
            self.testing_path = 'datasets/ED_size12_full_test.csv'
            self.validation_path = 'datasets/ED_size12_full_validation.csv'

    def train_loader(self, batch_size = None, shuffle = False):

        if batch_size is None:
            batch_size = self.batch

        train_data_source = np.array(pd.read_csv(self.training_path, header=None))
        train_samples_no = train_data_source.shape[0]
        train_labels = train_data_source[:,0]
        train_data = np.delete(train_data_source, 0, 1) # delete first column (0 element in 1 axis)

        # Shuffling ordered data, but preserving the mask (since we want to remember for which U/t the datapoint was calculated)
        mask = np.arange(train_samples_no)
        np.random.seed(0)
        np.random.shuffle(mask)

        np.save('model/CNN_LLvsCDWI_l005_mask.npy', mask)

        masked_train_data = train_data[mask]
        masked_train_labels = train_labels[mask]

        train_set = NumpyToPyTorch_DataLoader(masked_train_data, masked_train_labels)

        train_loader = DataLoader(train_set,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 0,
                          pin_memory = False # CUDA only, this lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer from CPU to GPU during training
                         )
        return train_loader#, mask - it's always the same, thanks to the fixed random seed, so I needed to save it only once

    def test_loader(self, batch_size = None):

        if batch_size is None:
            batch_size = self.batch

        test_data_source = np.array(pd.read_csv(self.testing_path, header=None)) # w/o 'header=None it ignores the first row
        self.test_samples_no = test_data_source.shape[0]
        self.test_labels = test_data_source[:,0]
        self.test_data = np.delete(test_data_source, 0, 1) # delete first column (0 element in 1 axis)

        test_set = NumpyToPyTorch_DataLoader(self.test_data, self.test_labels)

        test_loader = DataLoader(test_set,
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = 0,
                        pin_memory=True # CUDA only
                        )
        return test_loader

    def single_test_loader(self, no_sample):
        test_data_source = np.array(pd.read_csv(self.testing_path, header=None)) # w/o 'header=None it ignores the first row
        self.test_samples_no = 1
        all_test_labels = test_data_source[:,0]
        all_test_data = np.delete(test_data_source, 0, 1) # delete first column (0 element in 1 axis)

        self.test_data = all_test_data[no_sample]
        self.test_labels = all_test_labels[no_sample]

        test_set = SingleNumpyToPyTorch_DataLoader(self.test_data, self.test_labels)

        single_test_loader = DataLoader(test_set,
                        batch_size = 1,
                        shuffle = False,
                        num_workers = 1,
                        #pin_memory=True # CUDA only
                        )
        return single_test_loader

    def validation_loader(self, batch_size = None):

        if batch_size is None:
            batch_size = self.batch

        validation_data_source = np.array(pd.read_csv(self.validation_path, header=None)) # w/o 'header=None it ignores the first row
        self.validation_samples_no = validation_data_source.shape[0]
        self.validation_labels = validation_data_source[:,0]
        self.validation_data = np.delete(validation_data_source, 0, 1) # delete first column (0 element in 1 axis)

        validation_set = NumpyToPyTorch_DataLoader(self.validation_data, self.validation_labels)

        validation_loader = DataLoader(validation_set,
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = 0,
                        #pin_memory=True # CUDA only
                        )
        return validation_loader

    def training_samples_no(self):
        train_data_source = np.array(pd.read_csv(self.training_path, header=None))
        train_samples_no = train_data_source.shape[0]
        return train_samples_no
