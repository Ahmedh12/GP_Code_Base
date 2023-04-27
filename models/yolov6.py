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

from layers.common import  RepVGGBlock, RepBlock, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, ConvWrapper 
from layers.common import  SimConv , BiFusion
from layers.common import  Conv
from layers.common import  get_block
from utils.utils import generate_anchors , dist2bbox , initialize_weights
import math
import torch.nn as nn
import torch
import torch.nn.functional as F


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
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
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
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
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

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])

    def initialize_biases(self):

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
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
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)


            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
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
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + num_anchors),
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
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + num_anchors),
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
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        )
    )

    if num_layers == 4:
        head_layers.add_module('stem3',
            # stem3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=1,
                stride=1
            )
        )
        head_layers.add_module('cls_conv3',
            # cls_conv3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('reg_conv3',
            # reg_conv3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('cls_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('reg_pred3',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=4 * (reg_max + num_anchors),
                kernel_size=1
            )
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
    fuse_P2 = False
    cspsppf = True
    num_repeat_neck = [12, 12, 12, 12]
    channels_list_neck = [256, 128, 128, 256, 256, 512]
    use_dfl = False
    reg_max = 0
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
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def build_model(num_classes, device, fuse_ab=False, distill_ns=False):
    model = Model(channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
    return model

#########################################################################Module Test#########################################################################
def testYolov6():
    from torch import load
    from torchsummary import summary

    model = build_model(num_classes=5, device='cpu',fuse_ab=False, distill_ns=False)
    print(*list(model.state_dict().keys()),sep="\n")
    print(len(list(model.state_dict().keys())))

    # model.load_state_dict(load(r'C:\Users\sandr\Desktop\ahm\deyolo_refactored\models\weights\YOLOv6_weights2.pt', map_location='cpu'))
    model_dict = load(r'C:\Users\sandr\Desktop\ahm\deyolo_refactored\models\weights\YOLOv6_weights2.pt', map_location='cpu')
    # print(*list(model_dict.keys()),sep="\n")
    # print(len(list(model_dict.keys())))


if __name__ == '__main__':
    testYolov6()
