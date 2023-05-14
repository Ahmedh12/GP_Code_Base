from models.utils.datasets import createRTTSDataLoader
from models.TransWeather import init_TransWeather
from models.yolov6 import init_Yolov6
from models.yolov3 import init_Yolov3
from models.DENet import init_DENet
from evaluate import evaluate
import torch
import torch.nn as nn

import os

#0 -> test
#1 -> val

class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person'] #class names RTTS dataset

root_dir='../datasets/RTTS/images/val_large/'
num_workers=min(8, os.cpu_count()-1)
batch_size=16
image_size=416
shuffle=False
device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
#create dataloader
dataloader = createRTTSDataLoader(root_dir= root_dir, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle , image_size= image_size)

conf_thres = 0.001
nms_thres = 0.35

class DEYOLOv3(nn.Module):
    def __init__(self):
        super(DEYOLOv3, self).__init__()
        self.yolo = init_Yolov3()
        self.de = init_DENet()
    def forward(self, x):
        x = self.de(x)
        x = self.yolo(x)
        return x

class DEYOLOv6(nn.Module):
    def __init__(self):
        super(DEYOLOv6, self).__init__()
        self.yolo = init_Yolov6()
        self.de = init_DENet()
    def forward(self, x):
        x = self.de(x)
        x = self.yolo(x)
        return x

class TransWeatherYOLOv3(nn.Module):
    def __init__(self):
        super(TransWeatherYOLOv3, self).__init__()
        self.yolo = init_Yolov3()
        self.transweather = init_TransWeather()
    def forward(self, x):
        x = self.transweather(x)
        x = self.yolo(x)
        return x
    
class TransWeatherYOLOv6(nn.Module):
    def __init__(self):
        super(TransWeatherYOLOv6, self).__init__()
        self.yolo = init_Yolov6()
        self.transweather = init_TransWeather()
    def forward(self, x):
        x = self.transweather(x)
        x = self.yolo(x)
        return x
    
class HybridEnhancedYOLOv6(nn.Module):
    def __init__(self):
        super(HybridEnhancedYOLOv6, self).__init__()
        self.yolo = init_Yolov6()
        self.de = init_DENet()
        self.transweather = init_TransWeather()

    def forward(self, x):
        x = self.de(x)
        x = self.transweather(x)
        x = self.yolo(x)
        return x

def test(model , model_name , inf_num = 0 , run_dir = './runs/'):
    print(f"Inference number {inf_num} on {model_name} model ...")
    run_dir = run_dir + model_name + '_inference_'+str(inf_num)
    #evaluate model
    evaluate(model,
            dataloader,
            class_names=class_names,
            img_size = image_size,
            device= device,
            conf_thres= conf_thres,
            nms_thres= nms_thres,
            run_dir= run_dir,
            verbose=True)
    

if __name__ == '__main__':
    inf_num = 2

    model_names = ["YOLOv3" , "YOLOv6" , "DEYOLOv3" , "DEYOLOv6" , "TransWeatherYOLOv3" , "TransWeatherYOLOv6" , "HybridEnhancedYOLOv6"]
    models = [init_Yolov3() , init_Yolov6() , DEYOLOv3() , DEYOLOv6() , TransWeatherYOLOv3() , TransWeatherYOLOv6() , HybridEnhancedYOLOv6()]
    
    # for name , model in zip(model_names , models):
    #     model.to(device)
    #     test(model , name , inf_num = inf_num)

    models[6].to(device)
    test(models[6] , model_names[6] , inf_num = 1)
    



    
