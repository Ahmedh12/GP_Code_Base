from models.deyolo import DENet
from utils.datasets import createRTTSDataLoader
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

def test_DENet():
    model = DENet()
    #load the weights
    model.load_state_dict(torch.load('weights/enhancement_weights.pt', map_location='cpu'))
    model.eval()

    # load the data
    dataLoader = createRTTSDataLoader(batch_size=4, num_workers=0, image_size=544 , shuffle=True)
    for batch_i, (imgs, labels) in enumerate(dataLoader):
        if batch_i == 10:
            break
        imgs_enhanced = model(imgs)

        #convert to numpy
        imgs_enhanced = imgs_enhanced.permute(0,2,3,1).detach().numpy() # (B, H, W, C)
        imgs = imgs.permute(0,2,3,1).detach().numpy() # (B, H, W, C)

        #Show the images
        for i in range(imgs.shape[0]):
            cocatenated = np.concatenate((imgs[i], imgs_enhanced[i]), axis=1)
            cv2.imshow('Original vs. Enhanced Image', cocatenated)
            cv2.waitKey(2)

if __name__ == '__main__':

    #shows the images and their enhanced versions in the first batch
    test_DENet()
















    # data_dict = {'val': '../datasets/RTTS/images/val/', 
    #              'nc': 5, 
    #              'is_coco': False, 
    #              'names': ['bicycle', 'bus', 'car', 'motorbike', 'person'], 
    #              'name': 'rtts_5c'}
    # # opt = TestOptions().init()