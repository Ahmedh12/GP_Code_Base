from models.utils.datasets import createRTTSDataLoader
from models.TransWeather import init_TransWeather
from models.yolov6 import init_Yolov6
from models.yolov3 import init_Yolov3
from models.DENet import init_DENet
from evaluate import evaluate
import torch
import torch.nn as nn
import time
import os

#0 -> test
#1 -> val

class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person'] #class names RTTS dataset

root_dir='../datasets/RTTS/images/val/'
num_workers=min(8, os.cpu_count()-1)
batch_size=4
shuffle=False
device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
#create dataloader
dataloader = createRTTSDataLoader(root_dir= root_dir, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def testYolov6(inf_num = 0):
    print(f"Inference number {inf_num} on Yolov6 model ...")
    run_dir = './runs/Yolov6_inference_'+str(inf_num)

    #create model
    model = init_Yolov6().to(device)
    #evaluate model
    evaluate(model, 
            dataloader,
            class_names=class_names,
            img_size = 544, 
            device= device, 
            conf_thres=0.25, 
            nms_thres=0.40,
            run_dir= run_dir, 
            verbose=True)
    
def testYolov3(inf_num = 0):
    print(f"Inference number {inf_num} on Yolov3 model ...")
    run_dir = './runs/Yolov3_inference_'+str(inf_num)
    #create model
    model = init_Yolov3().to(device)
    #evaluate model
    evaluate(model, 
            dataloader,
            class_names=class_names,
            img_size = 544, 
            device= device, 
            conf_thres=0.25, 
            nms_thres=0.40,
            run_dir= run_dir, 
            verbose=True)
    
def testDEYOLOv3(inf_num = 0):
    print(f"Inference number {inf_num} on DEYOLOv3 model ...")
    class DEYOLOv3(nn.Module):
        def __init__(self):
            super(DEYOLOv3, self).__init__()
            self.yolo = init_Yolov3()
            self.de = init_DENet()
        def forward(self, x):
            x = self.de(x)
            x = self.yolo(x)
            return x
    
    run_dir = './runs/DEYOLOv3_inference_'+str(inf_num)
    #create model
    model = DEYOLOv3().eval().to(device)
    #evaluate model
    evaluate(model, 
            dataloader,
            class_names=class_names,
            img_size = 544, 
            device= device, 
            conf_thres=0.25, 
            nms_thres=0.40,
            run_dir= run_dir, 
            verbose=True)
     
def testDEYOLOv6(inf_num = 0):
    print(f"Inference number {inf_num} on DEYOLOv6 model ...")
    class DEYOLOv6(nn.Module):
        def __init__(self):
            super(DEYOLOv6, self).__init__()
            self.yolo = init_Yolov6()
            self.de = init_DENet()
        def forward(self, x):
            x = self.de(x)
            x = self.yolo(x)
            return x
        
    run_dir = './runs/DEYOLOv6_inference_'+str(inf_num)
    #create model
    model = DEYOLOv6().eval().to(device)
    #evaluate model
    evaluate(model,
            dataloader,
            class_names=class_names,
            img_size = 544,
            device= device,
            conf_thres=0.25,
            nms_thres=0.40,
            run_dir= run_dir,
            verbose=True)

def testTransWeatherYOLOv3(inf_num = 0):
    print(f"Inference number {inf_num} on TransWeatherYOLOv3  model ...")
    class TransWeatherYOLOv3(nn.Module):
        def __init__(self):
            super(TransWeatherYOLOv3, self).__init__()
            self.yolo = init_Yolov3()
            self.transweather = init_TransWeather()
        def forward(self, x):
            x = self.transweather(x)
            x = self.yolo(x)
            return x
    
    run_dir = './runs/TransWeatherYOLOv3_inference_'+str(inf_num)
    #create model
    model = TransWeatherYOLOv3().eval().to(device)
    #evaluate model
    evaluate(model,
            dataloader,
            class_names=class_names,
            img_size = 544,
            device= device,
            conf_thres=0.25,
            nms_thres=0.40,
            run_dir= run_dir,
            verbose=True)
    
def testTransWeatherYOLOv6(inf_num = 0):
    print(f"Inference number {inf_num} on TransWeatherYOLOv6  model ...")
    class TransWeatherYOLOv6(nn.Module):
        def __init__(self):
            super(TransWeatherYOLOv6, self).__init__()
            self.yolo = init_Yolov6()
            self.transweather = init_TransWeather()
        def forward(self, x):
            x = self.transweather(x)
            x = self.yolo(x)
            return x
    
    run_dir = './runs/TransWeatherYOLOv6_inference_'+str(inf_num)
    #create model
    model = TransWeatherYOLOv6().eval().to(device)
    #evaluate model
    evaluate(model,
            dataloader,
            class_names=class_names,
            img_size = 544,
            device= device,
            conf_thres=0.25,
            nms_thres=0.40,
            run_dir= run_dir,
            verbose=True)

def test(model , model_name , inf_num = 0 , run_dir = './runs/'):
    print(f"Inference number {inf_num} on {model_name} model ...")
    run_dir = run_dir + model_name + '_inference_'+str(inf_num)
    #evaluate model
    evaluate(model,
            dataloader,
            class_names=class_names,
            img_size = 544,
            device= device,
            conf_thres=0.25,
            nms_thres=0.40,
            run_dir= run_dir,
            verbose=True)
    

if __name__ == '__main__':
    # inf_num = int(time.localtime().tm_min)
    inf_num = 1
    testYolov3(inf_num = inf_num)
    testYolov6(inf_num = inf_num)
    testDEYOLOv3(inf_num = inf_num)
    testDEYOLOv6(inf_num = inf_num)
    testTransWeatherYOLOv3(inf_num = inf_num)
    testTransWeatherYOLOv6(inf_num = inf_num)
