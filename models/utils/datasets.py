from __future__ import print_function, division
import os
from skimage import io , transform
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode

class RTTS_Dataset(Dataset):
    """RTTS dataset."""

    def __init__(self, root_dir , transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            label_dir (string): Directory with all the labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        assert root_dir.find('images') != -1, 'root_dir should contain a folder called images' 
        self.label_dir = root_dir.replace('images', 'labels')
        self.transform = transform
        self.img_names = os.listdir(root_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.img_names[idx])
        label_name = os.path.join(self.label_dir,
                                self.img_names[idx].replace('.png', '.txt'))
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = open(label_name, 'r').readlines()
        label = [l.strip().split(' ') for l in label]
        label = [[int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in label]

        if self.transform:
            image = self.transform(image)
        return image , label
    
    def collate_fn(self, batch):
        images = []
        labels = []
        for b in batch:
            images.append(b[0])
            labels.append(b[1])
        images = torch.stack(images, dim=0) # (B, C, H, W)
        return images, labels
    
class Resize(object):
    """Resize ndarrays in sample to Tensors."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors. (H,W,C) -> (C,H,W))"""

    def __call__(self, image):
        return torch.from_numpy(image).permute(2,0,1).float() / 255.0

#############################################################Test Functions############################################################################
def test_RTTS_Dataset():
    #create a dataset object
    rtts_dataset = RTTS_Dataset(root_dir='../datasets/RTTS/images/val/', transform=transforms.Compose([ToTensor()]))
    #record time
    start = time.time()
    #show the first 4 images in a 2*2 grid
    for i in range(len(rtts_dataset)):
        image , label = rtts_dataset[i]
        image = image.permute(1,2,0)
        print(i, image.shape)
        ax = plt.subplot(2, 2, i + 1)
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        #show bounding boxes
        for l in label:
            x1 = (l[1] - l[3]/2) * image.shape[1]
            y1 = (l[2] - l[4]/2) * image.shape[0]
            x2 = (l[1] + l[3]/2) * image.shape[1]
            y2 = (l[2] + l[4]/2) * image.shape[0]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=1)
            ax.add_patch(rect)

        plt.imshow(image)
        if i == 3:
            print('Time taken to load 4 images using for loop: {} seconds'.format(time.time() - start))
            plt.show()
            break


def test_RTTS_Dataloader():
    #create a dataset object
    rtts_dataset = RTTS_Dataset(root_dir='../datasets/RTTS/images/val/' , transform=transforms.Compose([Resize(544), ToTensor()]))
    #create a dataloader object
    dataloader = DataLoader(rtts_dataset, batch_size=4, shuffle=True, num_workers = 0 , collate_fn=rtts_dataset.collate_fn)
    #record time
    start = time.time()
    #show the first 4 batches of 4 images each batch in a  2*2 grid , and all batches in a 2*2 grid with bounding boxes using grid of subplots
    _ , axs = plt.subplots(nrows=4 , ncols=4 , figsize=(12,12))
    for i_batch, (images , labels) in enumerate(dataloader):
        for j in range(images.shape[0]):

            #show images
            ax = axs[i_batch][j]
            ax.set_title('Batch #{} , Sample #{}'.format(i_batch , j))
            ax.axis('off')
            ax.imshow(images[j].permute(1,2,0))

            #show bounding boxes
            for l in labels[j]:
                x1 = (l[1] - l[3]/2) * images[j].shape[1]
                y1 = (l[2] - l[4]/2) * images[j].shape[2]
                x2 = (l[1] + l[3]/2) * images[j].shape[1]
                y2 = (l[2] + l[4]/2) * images[j].shape[2]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=1)
                ax.add_patch(rect)
            
        if i_batch == 3:
            print('Time taken to load 4 batches of 4 images using dataloader: {} seconds'.format(time.time() - start))
            plt.show()
            break

##################################################################Export Functions##################################################################
def createRTTSDataLoader(root_dir = '../datasets/RTTS/images/val/' , batch_size = 4 , num_workers = 0 , image_size = 544 , shuffle = True):
    #create a dataset object
    rtts_dataset = RTTS_Dataset(root_dir=root_dir , transform=transforms.Compose([Resize(image_size) , ToTensor()]))
    #create a dataloader object
    dataloader = DataLoader(rtts_dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers , collate_fn=rtts_dataset.collate_fn)
    return dataloader



#################################################################MAIN###############################################################################
if __name__ == '__main__':
    test_RTTS_Dataset()
    test_RTTS_Dataloader()

