if __name__ == '__main__':
    from DENet import DENet
    from models.yolov3 import YOLO_BASE , Darknet53, Neck, Detect
else:
    from .DENet import DENet
    from .yolov3 import YOLO_BASE , Darknet53, Neck, Detect

class DEYOLO(YOLO_BASE):
    def __init__(self, class_names, hyp, verbose=False):
        super().__init__(class_names, hyp, verbose)
        # enhancement
        self.enhancement = DENet()

        # backbone
        self.backbone = Darknet53()
        # Neck
        self.neck = Neck()
        # head
        self.Detect = Detect(nc=self.num_classes)

        # Build strides, anchors
        self.build_strides_anchors()

        # init model weight
        self.init_weight()

        # self.freeze_parms()

        # print model info
        self.model_info(verbose=False)

    def freeze_parms(self, backbone=True, neck=True, head=True):
        for v in self.backbone.parameters():
            v.requires_grad = False
        for v in self.neck.parameters():
            v.requires_grad = False
        for v in self.Detect.parameters():
            v.requires_grad = False

    def forward(self, x):
        x = self.enhancement(x)

        C3, C4, C5 = self.backbone(x)

        # neck
        C3, C4, C5 = self.neck([C3, C4, C5])
        # head
        yolo_out = self.Detect([C3, C4, C5])

        # return
        return yolo_out
