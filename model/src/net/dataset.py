import torch
from torch.utils.data import Dataset

from .preprocess import *

class Flight_Dataset(Dataset):
    """
    Flight_Dataset Object
    """
    def __init__(self, dataset, data_labels, labels_dct, mode):
        """
        Initialises the Flight_Dataset object.
        :param dataset: numpy array of the dataset (in function_class.py)
        :param data_labels: labels of the dataset given
        :param labels_dct: dictionary containing encoding of classes
        :param mode: train or test; assign encodings of labels
        """
        self.dataset = dataset
        self.data_labels = data_labels
        self.labels_dct = labels_dct
        self.mode = mode

    def __len__(self):
        '''
        Get the length of dataset.
        '''
        return self.dataset.__len__()

    def __getitem__(self, index):
        '''
        Get the item for each batch
        :return: a tuple of 6 object:
        1) normalized features of dataset
        2) labels of dataset (one-hot encoded and labels_dct)
        3) labels of dataset (encoded with labels_dct)
        4) length of each sequences without padding
        5) track id of each row in the dataset
        '''
        features, length, trajid = preprocess_x(self.dataset[index])
        labels, label_index = preprocess_y(self.data_labels[index], self.labels_dct, self.mode)
        return (torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(label_index), length, trajid)