#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.utils.events import LOGGER


class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False, distill_ns=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        # x:tensor = (32, 3, 640, 640) [0,1]
        export_mode = torch.onnx.is_in_onnx_export()
        x = self.backbone(x) # x: tuple = (tensor0, tensor1, tensor2, tensor3)     tensor0:tensor = (32, 64, 160, 160);   tensor1:tesnor = (32, 128, 80, 80);    tensor2:tensor = (32, 256, 40, 40);    tensor3:tensor = (32, 512, 20, 20) csdn图中只有tensor123对应下采样8,16,32倍
        x = self.neck(x) # x:list = [tensor0, tensor1, tensor2]  tensor0:tensor = (32, 64, 80, 80);   tensor1:tensor = (32, 128, 40, 40);   tensor2:tensor = (32, 256, 20, 20)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x) # featmaps = [tensor0, tensor1, tensor2]
        x = self.detect(x) # x:tuple = (list, tensor1, tensor2)  list=[tensor00, tesnor01, tensor02] tensor00=(32, 64, 80, 80)  tesnor01=(32, 128, 40, 40) tensor02=(32, 256, 20, 20) ;         tensor1=(32, 8400, 1)   tensor2 = (32, 8400, 4)
        # 也即[(N, 64, 80, 80), (N, 128, 40, 40), (N, 256, 20, 20)] ,  cls_score（N, 8400, 1）,   reg_distri (N, 8400, 4 + 12)
        return x if export_mode is True else [x, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)] # [32, 64, 128, 256, 512, 128, 64, 64, 128, 128, 256]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e
        )
    else:
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
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    else:
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers) # channels_list=[32, 64, 128, 256, 512, 128, 64, 64, 128, 128, 256]
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False):
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
    return model
