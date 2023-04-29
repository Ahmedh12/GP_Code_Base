import os

#prepare RTTS dataset labels and create test,train,val splits
#https://drive.google.com/file/d/16xuZv5KKGLm-k4qgi-MRkrdYQxhQZrWR/view?usp=share_link
 
classes_dict={
    "bicycle":0,
    "bus":1,
    "car":2,
    "motorbike":3,
    "person":4}

def createLabelFile():
    import xml.etree.ElementTree as ET
    import shutil
    import random



    paths = [
        './filename',
        './size/height',
        './size/width',
        './object/name',
        './object/bndbox/xmin',
        './object/bndbox/xmax',
        './object/bndbox/ymin',
        './object/bndbox/ymax',
    ]


    dest_dir = "custom_dataset\\"

    global train_count , test_count , val_count
    train_count = 0 #2600
    test_count = 0 #861
    val_count = 0 #861

    def  createLabelFileHelper(xml_filePath,dir):
        img_dir = ""
        label_dir = ""
        img_orig_dir = "custom_dataset\RTTS\JPEGImages\\"
        
        global train_count , test_count , val_count

        choice = int(random.random() * 3) #0

        if choice == 0 and train_count <=2600:
            img_dir = dir+"images\\train"
            label_dir = dir+"labels\\train\\"
            train_count+=1
            print("train_count: "+str(train_count))
        elif  choice == 1 and test_count <= 861:
            img_dir = dir+"images\\test"
            label_dir = dir+"labels\\test\\"
            test_count+=1
            print("test_count: "+str(test_count))
        elif  choice == 2 and val_count <= 861:
            img_dir = dir+"images\\val"
            label_dir = dir+"labels\\val\\"
            val_count+=1
            print("val_count: "+str(val_count))
        else:
            img_dir = dir+"images\\train"
            label_dir = dir+"labels\\train\\"
            train_count+=1
            print("train_count: "+str(train_count))


        obj_classes = []
        xmin , xmax , ymin , ymax = [] , [] , [] , []

        tree = ET.parse(xml_filePath)
        root = tree.getroot()
        for path in paths:
            for itm in root.findall(path):
                if(path.split("/")[-1] == 'name'):
                    obj_classes.append(classes_dict[itm.text])
                elif path.split("/")[-1] == 'xmin':
                    xmin.append(int(itm.text))
                elif path.split("/")[-1] == 'xmax':
                    xmax.append(int(itm.text))
                elif path.split("/")[-1] == 'ymin':
                    ymin.append(int(itm.text))
                elif path.split("/")[-1] == 'ymax':
                    ymax.append(int(itm.text))
                elif path.split("/")[-1] == 'height' :
                    img_height = int(itm.text)
                elif path.split("/")[-1] == 'width' :
                    img_width = int(itm.text)
                elif path.split("/")[-1] == 'filename' :
                    fileName = itm.text

        #Normalize the values
        x_center = [(x + (xmax[i] - x)/2)/img_width for i,x in enumerate(xmin)]
        y_center = [(y + (ymax[i] - y)/2)/img_height for i,y in enumerate(ymin)]
        width = [(xmax[i] - x)/img_width for i,x in enumerate(xmin)] #width
        height = [(ymax[i] - y)/img_height for i,y in enumerate(ymin)] #height

        try:
            shutil.move(img_orig_dir+fileName,img_dir)
            file  = open(label_dir+fileName.split(".")[0]+".txt","w")
            for i in range(len(obj_classes)):
                file.write(str(obj_classes[i])+" ")
                file.write(str(x_center[i])+" ")
                file.write(str(y_center[i])+" ")
                file.write(str(width[i])+" ")
                file.write(str(height[i])+" ")
                file.write("\n")
            file.close()
        except OSError as e:
            return

    xmlFiles = os.listdir("custom_dataset\RTTS\Annotations")
    # createLabellFile("custom_dataset\RTTS\Annotations\\"+xmlFiles[0],"custom_dataset\\")
    for xmlFile in xmlFiles:
        print("Processing: "+xmlFile)
        createLabelFileHelper("custom_dataset\RTTS\Annotations\\"+xmlFile,dest_dir)
        
    
    print("Done")

def testLabelFile():
    import cv2
    #class names
    classes = ["bicycle","bus","car","motorbike","person"]
    #load image
    img = cv2.imread(r"custom_dataset\images\test\AM_Bing_211.png")
    #load label
    file = open(r"custom_dataset\labels\test\AM_Bing_211.txt","r")
    lines = file.readlines()
    for line in lines:
        line = line.split(" ")
        # print(line)
        center_x = float(line[1]) * img.shape[1]
        center_y = float(line[2]) * img.shape[0]
        width = float(line[3]) * img.shape[1]
        height = float(line[4]) * img.shape[0]
        # print(xmin,ymin,xmax,ymax)
        xmin = int(center_x - width/2)
        xmax = int(center_x + width/2)
        ymin = int(center_y - height/2)
        ymax = int(center_y + height/2)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        cv2.putText(img,classes[int(line[0])],(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow("img",img)
    cv2.waitKey(0)

def init_dirs():
    if not os.path.exists("custom_dataset\images"):
        os.mkdir("custom_dataset\images")
        os.mkdir("custom_dataset\images\\train")
        os.mkdir("custom_dataset\images\\test")
        os.mkdir("custom_dataset\images\\val")
    if not os.path.exists("custom_dataset\labels"):
        os.mkdir("custom_dataset\labels")
        os.mkdir("custom_dataset\labels\\train")
        os.mkdir("custom_dataset\labels\\test")
        os.mkdir("custom_dataset\labels\\val")

if __name__ == "__main__":
    # init_dirs()
    # createLabelFile()
    testLabelFile()
