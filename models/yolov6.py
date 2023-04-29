#https://learnopencv.com/yolov6-object-detection/#What-is-YOLOv6? 

# YOLOv6 uses VFL and DFL as loss functions for classification and box regression, respectively.
# For this reason, the YOLOv6 models use reparameterized backbones. In reparameterization, the network structure changes during training and inference 

# Backbone of YOLOv6 : EfficientRep (Efficient Reparametarization) = RepBlock, RepConv, and CSPStackRep blocks. 
# Neck of YOLOv6 : Rep-PAN (Reparameterized Path Aggregation Network)
# Head of YOLOv6 : Decoupled Head

# Rep-PAN is similar to PAN in that it fuses features from multiple levels of the backbone network to create a more complete representation of the input image. 
# However, Rep-PAN introduces a reparameterization technique that allows the network to dynamically adjust its structure during training and inference.

# In Rep-PAN, the bottom-up pathway is the same as in PAN. The input image is passed through a backbone network to compute a set of feature maps at different 
# levels of the network. Each level of the network corresponds to a different spatial resolution of the input image.

# In the top-down pathway, Rep-PAN uses reparameterization to dynamically adjust the structure of the network. During training, the top-down pathway is initialized 
# with a fixed set of convolutional layers. However, during inference, the network adjusts the number of convolutional layers based on the input image, effectively
# "growing" or "shrinking" the network as needed.

# This reparameterization technique allows Rep-PAN to adapt to the complexity of the input image and improve the accuracy of object detection. Additionally, 
# Rep-PAN is more computationally efficient than traditional PAN because it uses fewer convolutional layers when processing simpler images.

# Overall, Rep-PAN is a powerful and flexible architecture for the neck component in object detection models. Its use of reparameterization allows 
# the network to dynamically adjust its structure to improve accuracy and efficiency.

# The Decoupled Head is a new head architecture for object detection models. It is designed to improve the accuracy of object detection models by
# decoupling the classification and box regression tasks. The Decoupled Head is composed of two separate subnetworks: a classification subnetwork
# and a box regression subnetwork. The classification subnetwork is responsible for predicting the class of each object in the input image.
# The box regression subnetwork is responsible for predicting the bounding box of each object in the input image.

#Loss functions
# The YOLOv6 model uses two loss functions: VFL and DFL. VFL is a variant of the focal loss function that is used to train the classification subnetwork.
# DFL is a variant of the smooth L1 loss function that is used to train the box regression subnetwork along with SIoU and GIoU.

try:
    from layers.common import  RepVGGBlock, RepBlock, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, ConvWrapper 
    from layers.common import  SimConv , BiFusion
    from layers.common import  Conv
    from layers.common import  get_block
    from utils.utils import generate_anchors , dist2bbox , initialize_weights
except ImportError:
    from .layers.common import  RepVGGBlock, RepBlock, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, ConvWrapper 
    from .layers.common import  SimConv , BiFusion
    from .layers.common import  Conv
    from .layers.common import  get_block
    from .utils.utils import generate_anchors , dist2bbox , initialize_weights

import time
import math
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import load

# Head , Neck , Backbone for YOLOv6s

class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        fuse_P2=False,
        cspsppf=False
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2

        self.stem = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=RepVGGBlock,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=RepVGGBlock,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=RepVGGBlock,
            )
        )

        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            RepVGGBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=RepVGGBlock,
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)

class RepBiFPANNeck(nn.Module):
    """RepBiFPANNeck Module
    """
    # [64, 128, 256, 512, 1024] 
    # [256, 128, 128, 256, 256, 512]
    
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None  

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[5]], # 512, 256
            out_channels=channels_list[5], # 256
        )
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            block=block
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[5], channels_list[6]], # 256, 128
            out_channels=channels_list[6], # 128
        )
      
        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            block=block
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            block=block
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            block=block
        )
        

    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs

class Detect(nn.Module):
    '''Efficient Decoupled Head for fusing anchor-base branches.
    '''
    def __init__(self, num_classes=80, anchors=None, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        self.anchors_init= ((torch.tensor(anchors) / self.stride[:,None])).reshape(self.nl, self.na, 2)

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds_af = nn.ModuleList()
        self.reg_preds_af = nn.ModuleList()
        self.cls_preds_ab = nn.ModuleList()
        self.reg_preds_ab = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*7
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds_af.append(head_layers[idx+3])
            self.reg_preds_af.append(head_layers[idx+4])
            self.cls_preds_ab.append(head_layers[idx+5])
            self.reg_preds_ab.append(head_layers[idx+6])

    def initialize_biases(self):

        for conv in self.cls_preds_af:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.cls_preds_ab:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds_af:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.reg_preds_ab:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        if self.training:
            device = x[0].device
            cls_score_list_af = []
            reg_dist_list_af = []
            cls_score_list_ab = []
            reg_dist_list_ab = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w

                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                reg_feat = self.reg_convs[i](reg_x)

                #anchor_base
                cls_output_ab = self.cls_preds_ab[i](cls_feat)
                reg_output_ab = self.reg_preds_ab[i](reg_feat)

                cls_output_ab = torch.sigmoid(cls_output_ab)
                cls_output_ab = cls_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2)
                cls_score_list_ab.append(cls_output_ab.flatten(1,3))

                reg_output_ab = reg_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2)
                reg_output_ab[..., 2:4] = ((reg_output_ab[..., 2:4].sigmoid() * 2) ** 2 ) * (self.anchors_init[i].reshape(1, self.na, 1, 1, 2).to(device))
                reg_dist_list_ab.append(reg_output_ab.flatten(1,3))

                #anchor_free
                cls_output_af = self.cls_preds_af[i](cls_feat)
                reg_output_af = self.reg_preds_af[i](reg_feat)

                cls_output_af = torch.sigmoid(cls_output_af)
                cls_score_list_af.append(cls_output_af.flatten(2).permute((0, 2, 1)))
                reg_dist_list_af.append(reg_output_af.flatten(2).permute((0, 2, 1)))
                
            
            cls_score_list_ab = torch.cat(cls_score_list_ab, axis=1)
            reg_dist_list_ab = torch.cat(reg_dist_list_ab, axis=1)
            cls_score_list_af = torch.cat(cls_score_list_af, axis=1)
            reg_dist_list_af = torch.cat(reg_dist_list_af, axis=1)
            
            return x, cls_score_list_ab, reg_dist_list_ab, cls_score_list_af, reg_dist_list_af

        else: # inference
            device = x[0].device
            cls_score_list_af = []
            reg_dist_list_af = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w

                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                reg_feat = self.reg_convs[i](reg_x)

                #anchor_free
                cls_output_af = self.cls_preds_af[i](cls_feat)
                reg_output_af = self.reg_preds_af[i](reg_feat)

                if self.use_dfl:
                    reg_output_af = reg_output_af.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output_af = self.proj_conv(F.softmax(reg_output_af, dim=1))

                cls_output_af = torch.sigmoid(cls_output_af)
                cls_score_list_af.append(cls_output_af.reshape([b, self.nc, l]))
                reg_dist_list_af.append(reg_output_af.reshape([b, 4, l]))
                
            cls_score_list_af = torch.cat(cls_score_list_af, axis=-1).permute(0, 2, 1)
            reg_dist_list_af = torch.cat(reg_dist_list_af, axis=-1).permute(0, 2, 1)
            

            #anchor_free
            anchor_points_af, stride_tensor_af = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            pred_bboxes_af = dist2bbox(reg_dist_list_af, anchor_points_af, box_format='xywh')
            pred_bboxes_af *= stride_tensor_af

            pred_bboxes = pred_bboxes_af
            cls_score_list = cls_score_list_af

            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):

    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # cls_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # stem1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # cls_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # stem2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # cls_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
    )

    return head_layers


#########################################################################YOLOv6#############################################################################
class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, channels=3, num_classes=None, fuse_ab=False, distill_ns=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = 3                      
        self.backbone, self.neck, self.detect = build_network( channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        featmaps = []
        featmaps.extend(x)
        x = self.detect(x)
        return [x, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    depth_mul = 0.33
    width_mul = 0.5
    num_repeat_backbone = [1, 6, 12, 18, 6]
    channels_list_backbone = [64, 128, 256, 512, 1024]
    fuse_P2 = True
    cspsppf = True
    num_repeat_neck = [12, 12, 12, 12]
    channels_list_neck = [256, 128, 128, 256, 256, 512]
    use_dfl = True
    reg_max = 16
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block("conv_silu")
    BACKBONE = eval("EfficientRep")
    NECK = eval("RepBiFPANNeck")

    backbone = BACKBONE(
        in_channels=channels,
        channels_list=channels_list,
        num_repeats=num_repeat,
        block=block,
        fuse_P2=fuse_P2,
        cspsppf=cspsppf
    )

    neck = NECK(
        channels_list=channels_list,
        num_repeats=num_repeat,
        block=block
    )

    if distill_ns:
        if num_layers != 3:
            print('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    elif fuse_ab:
        anchors_init = [[10,13, 19,19, 33,23], 
                      [30,61, 59,59, 59,119], 
                      [116,90, 185,185, 373,326]]
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    else:
        #will not work now , will fix later
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def build_model(num_classes, device, fuse_ab=False, distill_ns=False):
    model = Model(channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
    return model


########################################################################Inference Utils######################################################################
def fuse_conv_and_bn(conv, bn):
    '''Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/.'''
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def fuse_model(model):
    '''Fuse convolution and batchnorm layers of the model.'''
    for m in model.modules():
        if (type(m) is Conv or type(m) is SimConv) and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward
    return model

def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output

def box_convert(x):
    # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,thickness=tf, lineType=cv2.LINE_AA)

def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color

def draw_BB(img,img_src,det,class_names,hide_labels=False,hide_conf=False):
    '''
    Draw bounding boxes and labels on img_src
    :param img: the rescalled image used for detection
    :param img_src: the original image
    :param det: the detection result
    :param class_names: the class names as a list
    :param hide_labels: whether to hide the labels
    :param hide_conf: whether to hide the confidence
    :return: None
    '''
    img_with_boxes = img_src.copy()
    if len(det):
        det[:, :4] = rescale(img.shape[2:], det[:, :4], img_with_boxes.shape).round()     
        for *xyxy, conf, cls in reversed(det):
            class_num = int(cls)  # integer class
            label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
            plot_box_and_label(img_with_boxes, max(round(sum(img_with_boxes.shape) / 2 * 0.003), 2), xyxy, label, color = generate_colors(class_num, True))
    return img_with_boxes

##########################################################################Export Functions###################################################################
def init_Yolov6(path = r'models\weights\YOLOv6_weights.pt'):
    model = build_model(num_classes=5, device='cpu',fuse_ab=True, distill_ns=False) 
    model.load_state_dict(load(path, map_location='cpu'))
    
    #batchnorm fusion
    fuse_model(model).eval()

    # Reparametarization : switch to deploy mode 
    # YOLOv6s : 
    #   training : 768 layers -> 656 layers after batchnorm fusion
    #   inference  268 layers -> 156 layers after batchnorm fusion
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    return model
#########################################################################Module Test#########################################################################
def testYolov6():
    import warnings
    warnings.filterwarnings("ignore")

    from utils.yolo_utils import letterbox

    #Model Initialization
    model = init_Yolov6()
    
    # print("Model is {count} layers at inferene time after BNLs fusion and reparametarization  "
    #       .format(count = len(list(model.named_parameters())))) #156 layers
 
    
    #Model Data for inference
    stride = model.stride
    classNames = ['bicycle', 'bus', 'car', 'motorbike', 'person'] #class names RTTS dataset
    img_size = 640 #should be divisible by max stride

    #load an image , preprocess it then convert it to a tensor of size (B,C,H,W)
    img = cv2.imread(r'C:\Users\sandr\Desktop\GP_Code\datasets\RTTS\images\val\AM_Bing_211.png')
    img_original = img.copy()
    img = letterbox(img, new_shape=img_size, auto=True, stride=32)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = torch.from_numpy(np.ascontiguousarray(img))
    img = img.float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0) #add batch dimension
    #get the prediction from the model as tensor
    '''
    pred - > tensor of shape (B, summation for all strides(img_height/stride * img_width/stride), classes_count + 5)
    example :
    image size = 640,384
    number of classes = 5
    max stride = 32
    strides = [8, 16, 32]
    pred = tensor of shape (1, 640*384/8*8 + 640*384/16*16 + 640*384/32*32, 5+5) = (1,5040,10)
    '''
    pred = model(img)[0] #returns [predictions , feature maps] list
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    pred = pred.float()

    #postprocess the prediction e.g. perform non-max suppression
    det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)[0]
    #draw the bounding boxes on the image and display it
    img_with_boxes = draw_BB(img,img_original,det,classNames)
    cv2.imshow('image',img_with_boxes)
    cv2.waitKey(0)

if __name__ == '__main__':
    testYolov6()
