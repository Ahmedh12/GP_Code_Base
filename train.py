import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp

from models.utils.datasets import createRTTSDataLoader
from models.TransWeather import init_TransWeather
from models.yolov6 import init_Yolov6
from models.yolov3 import init_Yolov3
from models.DENet import init_DENet

from tqdm import tqdm

import os

class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person'] #class names RTTS dataset

root_dir='../datasets/RTTS/images/val_large/'
num_workers=min(8, os.cpu_count()-1)
batch_size=16
image_size=320
shuffle=False
device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
#create dataloader
dataloader = createRTTSDataLoader(root_dir= root_dir, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle , image_size= image_size)

conf_thres = 0.001
nms_thres = 0.5


class TransWeatherYOLOv6(nn.Module):
    def __init__(self):
        super(TransWeatherYOLOv6, self).__init__()
        self.yolo = init_Yolov6()
        self.transweather = init_TransWeather()

    def forward(self, x):
        x = self.transweather(x)
        x = self.yolo(x)
        return x

class DEYOLOv6(nn.Module):
    def __init__(self):
        super(DEYOLOv6, self).__init__()
        self.yolo = init_Yolov6(train=True)
        self.de = init_DENet(train=True)
    
    def forward(self, x):
        x = self.de(x)
        pred , feature_maps = self.yolo(x)
        return pred , feature_maps
    
def train(model , device , train_loader , optimizer , epoch):
    model.train()
    for batch_idx, (data, target, _,  _) in enumerate(tqdm(train_loader , desc="Training")):
        data , target = data.to(device) , target.to(device)
        optimizer.zero_grad()
        with amp.autocast(enabled= device != 'cpu'):
            preds, s_featmaps = model(data)
            total_loss, loss_items = compute_loss((preds[0],preds[3],preds[4]), targets, epoch_num, step_num) # YOLOv6_af
            total_loss_ab, loss_items_ab = compute_loss_ab(preds[:3], targets, epoch_num, step_num) # YOLOv6_ab
            total_loss += total_loss_ab
            loss_items += loss_items_ab 
        
        # loss.backward()
        # optimizer.step()
        if batch_idx % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch , batch_idx * len(data) , len(train_loader.dataset) , 100. * batch_idx / len(train_loader) , loss.item()))


def main():
    model = DEYOLOv6().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters() , lr=0.001)
    for epoch in range(1, 10 + 1):
        train(model , device , dataloader , optimizer , epoch)
        torch.cuda.empty_cache()
        
    torch.save(model.state_dict() , "denet_yolov6.pth")

if __name__ == "__main__":
    main()
