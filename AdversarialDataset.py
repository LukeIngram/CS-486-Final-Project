# AdversarialDataset.py

import pickle
import os

import numpy as np

from torch.utils.data.dataset import Dataset


class AdversarialDataset(Dataset):

    """TODO DOCSTRING"""

    def __init__(self, data_root: str, transform = None):
        self.transform = transform
            
        imgs_path = os.path.join(data_root, 'imgs.pickle')
        labels_path = os.path.join(data_root, 'labels.pickle')

        
        with open (imgs_path, 'rb') as fp:
            self.images = pickle.load(fp)
            
        with open (labels_path, 'rb') as fp:
            self.labels = pickle.load(fp)
        fp.close()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index].float()
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, label)