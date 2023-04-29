# Deep Learning Object Detection Model Comparative Analysis

This repository contains a comparative analysis of multiple deep learning object detection models with image enhancement modules attached. The objective of this analysis is to evaluate the performance of these models in adverse Weather conditions.
 

## Methodology
```
                                TO BE ADDED
```

## Datasets

The following datasets were used for the comparative analysis:

- [RTTS](https://drive.google.com/file/d/16xuZv5KKGLm-k4qgi-MRkrdYQxhQZrWR/view?usp=share_link)

## Evaluation Metrics

The following evaluation metrics were used for the comparative analysis:

- mAP@0.5
- f1 score

## Models

### Detection Modules

The following deep learning object detection models were evaluated:

- [YOLOv3](https://arxiv.org/pdf/1804.02767.pdf)
- [YOLOv6](https://arxiv.org/pdf/2209.02976.pdf)
- [YOLOv7](https://arxiv.org/pdf/2207.02696.pdf)

### Image Enhancement Modules

The following image enhancement modules were evaluated:

- [DENet](https://openaccess.thecvf.com/content/ACCV2022/papers/Qin_DENet_Detection-driven_Enhancement_Network_for_Object_Detection_under_Adverse_Weather_ACCV_2022_paper.pdf)
- [TransWeather](https://arxiv.org/pdf/2111.14813.pdf)


## Results

The results of the comparative analysis are presented in the following format:

| Enhancement Module  | Detection Module   | mAP@0.5       | F1 Score | 
| --------------------| -------------------| --------------| ---------|
| Faster R-CNN        |                    |               |          | 
| Mask R-CNN          |                    |               |          | 
| YOLOv4              |                    |               |          | 
| SSD                 |                    |               |          | 
| RetinaNet           |                    |               |          | 

## Usage

To reproduce the results of the comparative analysis run the following commands:
```
    git clone https://github.com/Ahmedh12/GP_Code_Base.git
    cd GP_Code_Base
    pip install -r requirements.txt
    python test.py
```

## Contributer

[Ahmed Hussien](https://github.com/Ahmedh12)
[Khaled El-lethy](https://github.com/SandraHany)
[Millania Sameh](https://github.com/MillaniaSameh)
[Sandra Hany](https://github.com/SandraHany)

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.
