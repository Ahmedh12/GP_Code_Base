from __future__ import print_function, division
import os
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import numpy as np

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
        img_path = os.path.join(self.root_dir,
                                self.img_names[idx])
        label_path = os.path.join(self.label_dir,
                                self.img_names[idx].replace('.png', '.txt'))
        image = cv2.imread(img_path)
        image = [image]
        shape =  [image[0].shape[0], image[0].shape[1]]
        labels = open(label_path, 'r').readlines()
        labels = [l.strip().split(' ') for l in labels]
        labels = [[int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels]
        #list of lists to numpy array
        labels = np.array(labels)
        nL = len(labels)  # number of labels
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        if self.transform:
            image = self.transform(image[0])
        return image , labels_out , img_path , shape
    

    def collate_fn(self, batch):
        images , labels , paths , shapes =  zip(*batch)
        images = torch.cat(images, dim=0)
        for i , l in enumerate(labels):
            l[: , 0] = i
        labels = torch.cat(labels, dim=0)
        return images, labels , paths , shapes
    
class Resize(object):
    """Resize ndarrays in sample to Tensors."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors. (H,W,C) -> (C,H,W))"""

    def __call__(self, img):
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)  # ascontiguousarray
        img = torch.from_numpy(img).unsqueeze(0)
        return img.float().div(255.0)
        # return torch.from_numpy(image).permute(2,0,1).float() / 255.0

#############################################################Test Functions############################################################################
def test_RTTS_Dataset():
    #create a dataset object
    rtts_dataset = RTTS_Dataset(root_dir='../datasets/RTTS/images/val/', transform=transforms.Compose([ToTensor()]))
    #record time
    start = time.time()
    #show the first 4 images in a 2*2 grid
    for i in range(len(rtts_dataset)):
        image , labels , _ , _ = rtts_dataset[i]
        print(i, image.shape)
        image = image[0].permute(1,2,0)
        ax = plt.subplot(2, 2, i + 1)
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        #show bounding boxes

        for l in labels: # l = [0 , class , x , y , w , h]
            x1 = (l[2] - l[4]/2) * image.shape[1]
            y1 = (l[3] - l[5]/2) * image.shape[0]
            x2 = (l[2] + l[4]/2) * image.shape[1]
            y2 = (l[3] + l[5]/2) * image.shape[0]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=1)
            ax.add_patch(rect)

        plt.imshow(image)
        if i == 3:
            print('Time taken to load 4 images using for loop: {} seconds'.format(time.time() - start))
            plt.show()
            break


def test_RTTS_Dataloader(batch_size=4):
    #create a dataset object
    rtts_dataset = RTTS_Dataset(root_dir='../datasets/RTTS/images/val/' , transform=transforms.Compose([Resize(544), ToTensor()]))
    #create a dataloader object
    dataloader = DataLoader(rtts_dataset, batch_size=batch_size, shuffle=True, num_workers = 0 , collate_fn=rtts_dataset.collate_fn)
    #record time
    start = time.time()
    #show the first 4 batches of 4 images each batch in a  2*2 grid , and all batches in a 2*2 grid with bounding boxes using grid of subplots
    _ , axs = plt.subplots(nrows=4 , ncols=4 , figsize=(12,12))
    for i_batch, (images , labels , img_paths , sizes) in enumerate(dataloader):
        
        for j in range(batch_size):
            #show images
            ax = axs[i_batch][j]
            ax.set_title('Batch #{} , Sample #{}'.format(i_batch , j))
            ax.axis('off')
            #get the image as the firsyt 3 channels of the tensor
            curr_image = images[j]
            ax.imshow(curr_image.permute(1,2,0))

            #show bounding boxes
            for l in labels[labels[:,0] == j][: , 1:]: # l = [class , x , y , w , h]
                x1 = (l[1] - l[3]/2) * curr_image.shape[1]
                y1 = (l[2] - l[4]/2) * curr_image.shape[2]
                x2 = (l[1] + l[3]/2) * curr_image.shape[1]
                y2 = (l[2] + l[4]/2) * curr_image.shape[2]
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

