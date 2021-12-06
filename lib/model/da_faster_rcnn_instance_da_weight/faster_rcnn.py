import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
    grad_reverse,
)
from torch.autograd import Variable
import time
import pdb

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, lc, gc):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

    def forward(self, im_data, im_info, gt_boxes, num_boxes, phase=1, target=False, eta=1.0):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        if phase == 1:
            base_feat1 = self.RCNN_base1(im_data)
            # feed image data to base model to obtain base feature map
            base_feat = self.RCNN_base2(base_feat1)
            # domain invariant
            base_feat_di = self.di(base_feat)
            # domain specific
            base_feat_ds = base_feat - base_feat_di

            if target == False:
                d_pixel, _ = self.netD_pixel(grad_reverse(base_feat1))
                _, feat_pixel = self.netD_pixel(base_feat1.detach())

                domain_p_base, _ = self.netD_base(grad_reverse(base_feat))
                _,feat_base = self.netD_base(base_feat.detach())
                domain_p_ds,_ = self.netD_ds(grad_reverse(base_feat_ds))
                _,feat_ds = self.netD_ds(base_feat_ds.detach())

                # feed base feature map tp RPN to obtain rois
                self.RCNN_rpn.train()
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_di, im_info, gt_boxes, num_boxes)
                # if it is training phrase, then use ground trubut bboxes for refining
                if self.training:
                    roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                    rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                    rois_label = Variable(rois_label.view(-1).long())
                    rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                    rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                    rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
                else:
                    rois_label = None
                    rois_target = None
                    rois_inside_ws = None
                    rois_outside_ws = None
                    rpn_loss_cls = 0
                    rpn_loss_bbox = 0
                rois = Variable(rois)

                if cfg.POOLING_MODE == 'align':
                    pooled_feat_di = self.RCNN_roi_align(base_feat_di, rois.view(-1, 5))
                    pooled_feat = self.RCNN_roi_align(base_feat, rois.detach().view(-1, 5))

                cls_prob_list = []
                bbox_pred_list = []
                RCNN_loss_cls_list = []
                RCNN_loss_bbox_list = []

                pooled_elem = pooled_feat_di
                pooled_feat_di_out = self._head_to_tail_di(pooled_elem)

                if self.lc:
                    feat_pixel_di = feat_pixel.view(1, -1).repeat(pooled_feat_di_out.size(0), 1)
                    pooled_feat_di_out = torch.cat((feat_pixel_di, pooled_feat_di_out), 1)
                if self.gc:
                    feat = feat_ds
                    feat_n = feat.view(1, -1).repeat(pooled_feat_di_out.size(0), 1)
                    pooled_feat_di_out = torch.cat((feat_n, pooled_feat_di_out), 1)
                bbox_pred_di = self.RCNN_bbox_pred_di(pooled_feat_di_out)
                if self.training and not self.class_agnostic:
                    bbox_pred_view = bbox_pred_di.view(bbox_pred_di.size(0), int(bbox_pred_di.size(1) / 4), 4)
                    bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                    bbox_pred_di = bbox_pred_select.squeeze(1)

                cls_score_di = self.RCNN_cls_score_di(pooled_feat_di_out)
                cls_prob_di = F.softmax(cls_score_di, 1)

                RCNN_loss_cls_di = 0
                RCNN_loss_bbox_di = 0
                if self.training:
                    RCNN_loss_cls_di = F.cross_entropy(cls_score_di, rois_label)
                    RCNN_loss_bbox_di = _smooth_l1_loss(bbox_pred_di, rois_target, rois_inside_ws, rois_outside_ws)
                cls_prob_di = cls_prob_di.view(batch_size, rois.size(1), -1)
                bbox_pred_di = bbox_pred_di.view(batch_size, rois.size(1), -1)

                pooled_elem = pooled_feat
                pooled_feat_base_out = self._head_to_tail_base(pooled_elem)
                if self.lc:
                    feat_pixel_base = feat_pixel.view(1, -1).repeat(pooled_feat_base_out.size(0), 1)
                    pooled_feat_base_out = torch.cat((feat_pixel_base, pooled_feat_base_out), 1)
                if self.gc:
                    feat = feat_base
                    feat_n = feat.view(1, -1).repeat(pooled_feat_base_out.size(0), 1)
                    pooled_feat_base_out = torch.cat((feat_n, pooled_feat_base_out), 1)
                bbox_pred_base = self.RCNN_bbox_pred_base(pooled_feat_base_out)
                if self.training and not self.class_agnostic:
                    bbox_pred_view = bbox_pred_base.view(bbox_pred_base.size(0), int(bbox_pred_base.size(1) / 4), 4)
                    bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                    bbox_pred_base = bbox_pred_select.squeeze(1)
                cls_score_base = self.RCNN_cls_score_base(pooled_feat_base_out)
                cls_prob_base = F.softmax(cls_score_base, 1)

                RCNN_loss_cls_base = 0
                RCNN_loss_bbox_base = 0
                if self.training:
                    RCNN_loss_cls_base = F.cross_entropy(cls_score_base, rois_label)
                    RCNN_loss_bbox_base = _smooth_l1_loss(bbox_pred_base, rois_target, rois_inside_ws, rois_outside_ws)
                cls_prob_base = cls_prob_base.view(batch_size, rois.size(1), -1)
                bbox_pred_base = bbox_pred_base.view(batch_size, rois.size(1), -1)

                cls_prob_list.append(cls_prob_di)
                bbox_pred_list.append(bbox_pred_di)
                RCNN_loss_cls_list.append(RCNN_loss_cls_di)
                RCNN_loss_bbox_list.append(RCNN_loss_bbox_di)

                cls_prob_list.append(cls_prob_base)
                bbox_pred_list.append(bbox_pred_base)
                RCNN_loss_cls_list.append(RCNN_loss_cls_base)
                RCNN_loss_bbox_list.append(RCNN_loss_bbox_base)

                return rois, cls_prob_list, bbox_pred_list, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls_list, RCNN_loss_bbox_list, rois_label, d_pixel, domain_p_base, domain_p_ds
            else:
                d_pixel, _ = self.netD_pixel(grad_reverse(base_feat1))
                domain_p_base, _ = self.netD_base(grad_reverse(base_feat))
                domain_p_ds, _ = self.netD_ds(grad_reverse(base_feat_ds))
                return d_pixel, domain_p_base, domain_p_ds
        elif phase == 2:
            base_feat1 = self.RCNN_base1(im_data)
            # feed image data to base model to obtain base feature map
            base_feat = self.RCNN_base2(base_feat1)
            # domain invariant
            base_feat_di = self.di(base_feat)
            # domain specific
            base_feat_ds = base_feat - base_feat_di

            if target == False:
                _, feat_pixel = self.netD_pixel(base_feat1.detach())
                domain_p_ds,_ = self.netD_ds(grad_reverse(base_feat_ds))
                _,feat_ds = self.netD_ds(base_feat_ds.detach())
                self.RCNN_rpn.train()
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_di, im_info, gt_boxes, num_boxes)

                # if it is training phrase, then use ground trubut bboxes for refining
                if self.training:
                    roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                    rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                    rois_label = Variable(rois_label.view(-1).long())
                    rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                    rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                    rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
                else:
                    rois_label = None
                    rois_target = None
                    rois_inside_ws = None
                    rois_outside_ws = None
                    rpn_loss_cls = 0
                    rpn_loss_bbox = 0
                rois = Variable(rois)

                if cfg.POOLING_MODE == 'align':
                    pooled_feat_di = self.RCNN_roi_align(base_feat_di, rois.view(-1, 5))
                    pooled_feat_ds = self.RCNN_roi_align(base_feat_ds, rois.detach().view(-1, 5))
                    pooled_feat = self.RCNN_roi_align(base_feat, rois.detach().view(-1, 5))

                #Mutual Information
                Mutual_invariant = F.avg_pool2d(pooled_feat_di, (7, 7))[:,:,0,0]
                Mutual_specific = F.avg_pool2d(pooled_feat_ds, (7, 7))[:,:,0,0]
                Mutual_invariant = F.normalize(Mutual_invariant, dim=1)
                Mutual_specific = F.normalize(Mutual_specific, dim=1)
                Mutual_loss = Mutual_invariant * Mutual_specific
                Mutual_loss = torch.abs(torch.sum(Mutual_loss, dim=1))
                Mutual_loss = torch.mean(Mutual_loss)

                cls_prob_list = []
                bbox_pred_list = []
                RCNN_loss_cls_list = []
                RCNN_loss_bbox_list = []

                pooled_elem = pooled_feat_di
                pooled_feat_di_out = self._head_to_tail_di(pooled_elem)

                if self.lc:
                    feat_pixel_di = feat_pixel.view(1, -1).repeat(pooled_feat_di_out.size(0), 1)
                    pooled_feat_di_out = torch.cat((feat_pixel_di, pooled_feat_di_out), 1)
                if self.gc:
                    feat = feat_ds
                    feat_n = feat.view(1, -1).repeat(pooled_feat_di_out.size(0), 1)
                    pooled_feat_di_out = torch.cat((feat_n, pooled_feat_di_out), 1)
                bbox_pred_di = self.RCNN_bbox_pred_di(pooled_feat_di_out)
                if self.training and not self.class_agnostic:
                    bbox_pred_view = bbox_pred_di.view(bbox_pred_di.size(0), int(bbox_pred_di.size(1) / 4), 4)
                    bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                    bbox_pred_di = bbox_pred_select.squeeze(1)

                cls_score_di = self.RCNN_cls_score_di(pooled_feat_di_out)
                cls_prob_di = F.softmax(cls_score_di, 1)

                RCNN_loss_cls_di = 0
                RCNN_loss_bbox_di = 0
                if self.training:
                    RCNN_loss_cls_di = F.cross_entropy(cls_score_di, rois_label)
                    RCNN_loss_bbox_di = _smooth_l1_loss(bbox_pred_di, rois_target, rois_inside_ws, rois_outside_ws)
                cls_prob_di = cls_prob_di.view(batch_size, rois.size(1), -1)
                bbox_pred_di = bbox_pred_di.view(batch_size, rois.size(1), -1)

                cls_prob_list.append(cls_prob_di)
                bbox_pred_list.append(bbox_pred_di)
                RCNN_loss_cls_list.append(RCNN_loss_cls_di)
                RCNN_loss_bbox_list.append(RCNN_loss_bbox_di)

                return rois, cls_prob_list, bbox_pred_list, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls_list, RCNN_loss_bbox_list, rois_label, domain_p_ds, Mutual_loss
            else:
                domain_p_ds, _ = self.netD_ds(grad_reverse(base_feat_ds))
                self.RCNN_rpn.eval()
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_di, im_info, gt_boxes, num_boxes)
                if self.training:
                    roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                    rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                    rois_label = Variable(rois_label.view(-1).long())
                    rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                    rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                    rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
                else:
                    rois_label = None
                    rois_target = None
                    rois_inside_ws = None
                    rois_outside_ws = None
                    rpn_loss_cls = 0
                    rpn_loss_bbox = 0
                rois = Variable(rois)

                if cfg.POOLING_MODE == 'align':
                    pooled_feat_di = self.RCNN_roi_align(base_feat_di, rois.view(-1, 5))
                    pooled_feat_ds = self.RCNN_roi_align(base_feat_ds, rois.detach().view(-1, 5))
                    pooled_feat = self.RCNN_roi_align(base_feat, rois.detach().view(-1, 5))

                #Mutual Information
                Mutual_invariant = F.avg_pool2d(pooled_feat_di, (7, 7))[:,:,0,0]
                Mutual_specific = F.avg_pool2d(pooled_feat_ds, (7, 7))[:,:,0,0]
                Mutual_invariant = F.normalize(Mutual_invariant, dim=1)
                Mutual_specific = F.normalize(Mutual_specific, dim=1)
                Mutual_loss = Mutual_invariant * Mutual_specific
                Mutual_loss = torch.abs(torch.sum(Mutual_loss, dim=1))
                Mutual_loss = torch.mean(Mutual_loss)

                return domain_p_ds, Mutual_loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_di, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_di, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_base, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_base, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
