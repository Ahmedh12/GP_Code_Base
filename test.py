from evaluate import evaluate
from models.utils.datasets import createRTTSDataLoader
from models.DENet import init_DENet
from models.yolov6 import init_Yolov6
from models.yolov3 import init_Yolov3

def testYolov6(inf_num = 0):
    run_dir = './runs/Yolov6_inference_'+str(inf_num)
    class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person'] #class names RTTS dataset
    #create model
    model = init_Yolov6()
    #create dataloader
    dataloader = createRTTSDataLoader(batch_size=8, num_workers=2)
    #evaluate model
    evaluate(model, 
            dataloader,
            class_names=class_names,
            img_size = 544, 
            device='cpu', 
            conf_thres=0.25, 
            nms_thres=0.40,
            run_dir= run_dir, 
            verbose=True)
    
def testYolov3(inf_num = 0):
    run_dir = './runs/Yolov3_inference_'+str(inf_num)
    class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person'] #class names RTTS dataset
    #create model
    model = init_Yolov3()
    #create dataloader
    dataloader = createRTTSDataLoader(batch_size=8, num_workers=2)
    #evaluate model
    evaluate(model, 
            dataloader,
            class_names=class_names,
            img_size = 544, 
            device='cpu', 
            conf_thres=0.25, 
            nms_thres=0.40,
            run_dir= run_dir, 
            verbose=True)

if __name__ == '__main__':
    testYolov6()
    testYolov3()