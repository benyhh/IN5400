from sklearn.feature_extraction import img_to_graph
import torch
from torch.utils.data import Dataset
import os
import PIL.Image
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader

def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return sorted(classes), len(classes)

def get_labels(filename, n):
    labels = []
    count = 0
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            if count == n:
                return labels
            
            split = line.split(",")[1:]
            split = list(set(split[0].split()))
            labels.append(split)
            count += 1

    return labels


class ChannelSelect(torch.nn.Module):
    """This class is to be used in transforms.Compose when you want to use selected channels. e.g only RGB.
    It works only for a tensor, not PIL object.
    Args:
        channels (list or int): The channels you want to select from the original image (4-channel).

    Returns: img
    """
    def __init__(self, channels=[0, 1, 2]):
        super().__init__()
        self.channels = channels

    def forward(self, img):
        """
        Args:
            img (Tensor): Image
        Returns:
            Tensor: Selected channels from the image.
        """
        return img[self.channels, ...]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RainforestDataset(Dataset):
    def __init__(self, root_dir, trvaltest, transform = False):
        self.root_dir = root_dir
        self.trvaltest = trvaltest
        self.transform = transform
        classes, num_classes = get_classes_list()

        # TODO Binarise your multi-labels from the string. HINT: There is a useful sklearn function to
        # help you binarise from strings.
        lb = LabelBinarizer().fit(classes)
        self.imgfilenames = os.listdir(os.path.join(self.root_dir, "train-tif-v2/"))
        self.labels = []
        self.word_labels = get_labels(os.path.join(self.root_dir,"train_v2.csv"),len(self.imgfilenames))
        #print(np.sum(lb.transform(self.word_labels[0]), axis = 0))
        for label in self.word_labels:
            self.labels.append(np.sum(lb.transform(label),axis = 0))
        


        # TODO Perform a test train split. It's recommended to use sklearn's train_test_split with the following
        # parameters: test_size=0.33 and random_state=0 - since these were the parameters used
        # when calculating the image statistics you are using for data normalisation.
        X_train, X_test, y_train, y_test = train_test_split(self.imgfilenames, self.labels, test_size=0.33, random_state=321)
        #print(trvaltest, len(X_train), len(X_test))
        if self.trvaltest == 0:
            self.imgfilenames = X_train
            self.labels = y_train
        
        elif self.trvaltest == 1:
            self.imgfilenames = X_test
            self.labels = y_test
        
        
        #for debugging you can use a test_size=0.66 - this trains then faster
        

        # OR optionally you could do the test train split of your filenames and labels once, save them, and
        # from then onwards just load them from file.


    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        #get the label and filename, and load the image from file.
        imgname = self.imgfilenames[idx]
        
        img = PIL.Image.open(os.path.join(self.root_dir, imgname))    
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        sample = {'image': img,
                  'label': label,
                  'filename': imgname}
        return sample

