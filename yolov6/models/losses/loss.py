#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner
import sys

class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self, 
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5}
                 ):
        
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size
        
        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight       
        

    '''
     总体流程
     1. anchors生成
     2. 正负样本分配
     3. gt bbox 编码
     4. 计算各种loss
    '''
    def __call__(
        self,
        outputs,
        targets,
        epoch_num,
        step_num
    ):
        '''
        outputs =  preds = [(N, 64, 80, 80), (N, 128, 40, 40), (N, 256, 20, 20)] ,  (N, 8400, 1),  (N, 8400, 4) ltrb归一化之后的
        targets = (88, 6)  88个对象， [:, 0]是 序号  [:, 1]是得分   [:, 2:-1] 是 4个位置信息xywh（归一化之后的）
        '''  
        feats, pred_scores, pred_distri = outputs        

        ''' 
        Step1: 得到head的输出之后, 先生成anchor（都是原图尺度）
        anchors = (8400, 4)         anchor的xyxy            
        anchor_points = (8400, 2)   anchor的中心坐标
        n_anchors_list = [6400, 1600, 400]  
        stride_tensor = (8400, 1)
        '''
        anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
        '''with open("debug/demo01.txt", "w") as f:
            # torch.set_printoptions(edgeitems=sys.maxsize, threshold=sys.maxsize)
            f.write("anchors:\n"+str(anchors.tolist()))
            f.write("\n--------\n")
            f.write("anchor_points:\n"+str(anchor_points.tolist()))
            f.write("\n--------\n")
            f.write("n_anchors_list:\n"+str(n_anchors_list))
            f.write("\n--------\n")
            f.write("stride_tensor:\n"+str(stride_tensor.tolist()))'''
        '''
        print(anchors)
        print("------------------")
        print(anchor_points)
        print("------------------")
        print(n_anchors_list)
        print("------------------")
        print(stride_tensor)'''
        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.full((1,4), self.ori_img_size).type_as(pred_scores) # [[640., 640., 640., 640.]]
        batch_size = pred_scores.shape[0] # 32


        '''
        Step2: 根据targets得到gt信息
        '''
        # targets
        targets =self.preprocess(targets, batch_size, gt_bboxes_scale) 
        # （32, 6, 5） 每个image对应了一个(6, 5)的矩阵，其中6应该代表6个对象（根据一张图片中最大的对象的数量确定），5代表类别和xyxy     
        #  targets_xywh（归一化尺度，直接除以640）-> targets_xyxy原图尺度
        gt_labels = targets[:, :, :1] # 类别 （32, 6, 1）
        gt_bboxes = targets[:, :, 1:] # xyxy （32, 6, 4） 
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float() #（32, 6, 1）类似于子网掩码，只是标志该行位置是否有对象
        

        '''
        Step3: 根据head输出的预测 和(特征图尺度下的！)anchor_points_s 解码成预测边框（用于计算相关指标来正负样本分配？）。至此part 1部分完成
        '''
        # pboxes
        anchor_points_s = anchor_points / stride_tensor     # 这一步就是把原图尺度上的anchor_points(anchor中心点)除以下采样率得到预测特征图尺度上的anchor_points_s(一个格子大小为1,wh分别为80,40,20)
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri) 
        # 根据pred_distri（distance：ltrb）和anchor_points_s得到pred_bboxes (xyxy)
        # pred_bboxes:(32, 8400, 4) xyxy也是特征图尺度上的           pre_reg + anchor_points_s  ---decode---> pred_bbox
        '''with open("debug/demo01.txt", "w") as f:
            f.write("\n--------\n")
            f.write("pred_bboxes:\n"+str(pred_bboxes.tolist()))'''
        

        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        anchors,
                        n_anchors_list,
                        gt_labels,
                        gt_bboxes,
                        mask_gt,
                        pred_bboxes.detach() * stride_tensor)
            else:
                ''' 
                Step4: 这一步完成的是正负样本（前景背景样本分配），输出用于后续计算的labels、bboxes等。至此part 2部分完成，可以根据 part 1/2计算loss了
                
                得到如下：
                target_labels = (32, 8400) 全零
                target_bboxes = (32, 8400, 4) 原图尺度上
                target_scores = (32, 8400, 1) 大部分是0
                fg_mask = (32, 8400) bool类型的，标记哪个是前景样本  
                '''
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        pred_scores.detach(),
                        pred_bboxes.detach() * stride_tensor,
                        anchor_points,
                        gt_labels,
                        gt_bboxes,
                        mask_gt)
                 
                '''with open("debug/demo01.txt", 'w') as f:
                    f.write("\n--------\n")
                    f.write("target_labels:\n"+str(target_labels.tolist()))
                    f.write("\n--------\n")
                    f.write("target_bboxes:\n"+str(target_bboxes.tolist()))
                    f.write("\n--------\n")
                    f.write("target_scores:\n"+str(target_scores.tolist()))
                    f.write("\n--------\n")
                    f.write("fg_mask:\n"+str(fg_mask.tolist()))'''
                
        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        _anchors,
                        _n_anchors_list,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt,
                        _pred_bboxes * _stride_tensor)

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        _pred_scores,
                        _pred_bboxes * _stride_tensor,
                        _anchor_points,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
        #Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor # rescale到预测特征层尺度，每个grid cell长度是1，至此和pred_bbox对上了

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes)) # (32, 8400) 全零 -> 大多数是1
        '''with open("debug/demo01.txt", 'w') as f:
            f.write("target_labels:\n"+str(target_labels.tolist()))'''
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1] # (32, 8400, 1) 几乎全0
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)
            
        target_scores_sum = target_scores.sum()
		# avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson 
        if target_scores_sum > 0:
        	loss_cls /= target_scores_sum
        
        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)
        
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
       
        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0), 
                         (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                         (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()
     
    def preprocess(self, targets, batch_size, scale_tensor):# targets = (88, 6)  88个对象， [:, 0]是 序号  [:, 1]是类别   [:, 2:-1] 是 4个位置信息xywh
        targets_list = np.zeros((batch_size, 1, 5)).tolist() # (32, 1, 5)
        for i, item in enumerate(targets.cpu().numpy().tolist()): # item:(6)
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets #（32, 6, 5）

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
               
            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                        target_ltrb_pos) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
