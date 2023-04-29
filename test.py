from evaluate import evaluate
from models.utils.datasets import createRTTSDataLoader
from models.deyolo import DEYOLO
def test_DEYOLO():
    #create model
    class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person'] #class names RTTS dataset
    model = DEYOLO(class_names , hyp=None)
    #create dataloader
    dataloader = createRTTSDataLoader()
    #evaluate model
    evaluate(model, 
             dataloader,
             class_names=class_names,
             img_size = 544, 
             device='cpu', 
             conf_thres=0.25, 
             nms_thres=0.40, 
             verbose=True)

if __name__ == '__main__':
    test_DEYOLO()