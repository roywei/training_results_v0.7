# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
from torch.functional import F
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import GIoULoss
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_iou_batched
from maskrcnn_benchmark.modeling.positive_sampler import (
    PositiveSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from torch.nn.utils.rnn import pad_sequence
from maskrcnn_benchmark.layers import isr_p, carl_loss
from maskrcnn_benchmark.layers import CrossEntropyLoss
from mmcv.ops import nms_match


class PISALossOnePassComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
            self,
            proposal_matcher,
            fg_bg_sampler,
            box_coder,
            cls_agnostic_bbox_reg=False,
            decode=False,
            loss="SmoothL1Loss",
            carl=False,
            use_isr_p=False,
            use_isr_n=False,
            giou_box_weight=10.0,
            giou_carl_weight=10.0,
            batch_size_per_image=512,
            k=0.5,
            bias=0,
            score_threshold=0.05,
            iou_threshold=0.5
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.giou_loss = GIoULoss(eps=1e-6, reduction="mean", loss_weight=giou_box_weight)
        self.giou_loss_carl = GIoULoss(eps=1e-6, reduction="none", loss_weight=giou_carl_weight)
        self.cls_loss = CrossEntropyLoss()
        self.decode = decode
        self.loss = loss
        self.carl = carl
        self.use_isr_p = use_isr_p
        self.use_isr_n = use_isr_n
        self.k = k
        self.bias = bias
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.batch_size_per_image = batch_size_per_image

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def match_targets_to_proposals_batched(self, proposal, target):
        match_quality_matrix = boxlist_iou_batched(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix, batched=1)
        # Fast RCNN only need "labels" field for selecting the targets
        # how to do this for batched case?
        # target = target.copy_with_fields("labels")
        return matched_idxs

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        matched_idxs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets_per_image = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs_per_image = matched_targets_per_image.get_field("matched_idxs")

            labels_per_image = matched_targets_per_image.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_per_image == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image.masked_fill_(bg_inds, 0)

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_per_image == Matcher.BETWEEN_THRESHOLDS
            labels_per_image.masked_fill(ignore_inds, -1)  # -1 is ignored by sampler

            # compute regression targets
            # does not encode target if we need to decode pred later
            if not self.decode:
                regression_targets_per_image = self.box_coder.encode(
                    matched_targets_per_image.bbox, proposals_per_image.bbox
                )
            else:
                regression_targets_per_image = matched_targets_per_image.bbox

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_idxs.append(matched_idxs_per_image)
        return labels, regression_targets, matched_idxs

    def prepare_targets_batched(self, proposals, targets, target_labels):
        num_images = proposals.size(0)
        matched_idxs = self.match_targets_to_proposals_batched(proposals, targets)
        img_idx = torch.arange(num_images, device=proposals.device)[:, None]
        labels = target_labels[img_idx, matched_idxs.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels.masked_fill_(bg_inds, 0)
        ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels.masked_fill_(ignore_inds, -1)

        matched_targets = targets[img_idx, matched_idxs.clamp(min=0)]

        # does not encode target if we need to decode pred later
        if not self.decode:
            regression_targets = self.box_coder.encode(
                matched_targets.view(-1, 4), proposals.view(-1, 4)
            )
        else:
            regression_targets = matched_targets.view(-1, 4)
        return labels, regression_targets.view(num_images, -1, 4), matched_idxs

    def subsample(self, proposals, targets, features):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        num_images = len(proposals[0])
        target_boxes = pad_sequence([target.bbox for target in targets], batch_first=True, padding_value=-1)
        target_labels = pad_sequence([target.get_field("labels") for target in targets], batch_first=True,
                                     padding_value=-1)
        prop_boxes, prop_scores, image_sizes = proposals[0], proposals[1], proposals[2]
        labels, regression_targets, matched_idxs = self.prepare_targets_batched(prop_boxes, target_boxes, target_labels)

        # scores is used as a mask, -1 means box is invalid
        if num_images == 1:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, is_rpn=0, objectness=prop_scores)
            # when num_images=1, sampled pos inds only has 1 item, so avoid copy in torch.cat
            pos_inds_per_image = [torch.nonzero(sampled_pos_inds[0]).squeeze(1)]
            neg_inds_per_image = [torch.nonzero(sampled_neg_inds[0]).squeeze(1)]
        else:
            sampled_pos_inds, sampled_neg_inds, num_pos_samples, num_neg_samples = self.fg_bg_sampler(labels, is_rpn=0,
                                                                                                      objectness=prop_scores)
            pos_inds_per_image = sampled_pos_inds.split(list(num_pos_samples))
            neg_inds_per_image = sampled_neg_inds.split(list(num_neg_samples))
        prop_boxes = prop_boxes.view(-1, 4)
        regression_targets = regression_targets.view(-1, 4)
        labels = labels.view(-1)
        matched_idxs = matched_idxs.view(-1)
        result_proposals = []
        for i in range(num_images):
            inds = torch.cat([pos_inds_per_image[i], neg_inds_per_image[i]])
            box = BoxList(prop_boxes[inds], image_size=image_sizes[i])
            box.add_field("matched_idxs", matched_idxs[inds])
            box.add_field("regression_targets", regression_targets[inds])
            box.add_field("labels", labels[inds])
            result_proposals.append(box)
        self._proposals = result_proposals

        return result_proposals

    def isr_n(self, class_logits_batched, box_regression_batched):
        proposals = self._proposals
        num_images = len(proposals)

        labels_batched = [proposal.get_field("labels") for proposal in proposals]
        rois_batched = [a.bbox for a in proposals]
        regression_targets_batched = [proposal.get_field("regression_targets") for proposal in proposals]
        batch_sizes = [item.size(0) for item in labels_batched]
        class_logits_batched = class_logits_batched[0].split(batch_sizes)
        box_regression_batched = box_regression_batched[0].split(batch_sizes)

        # isr_n results
        sampled_inds_batched = []
        sampled_pos_inds_batched = []
        label_weights_batched = []
        box_weights_batched = []
        sampled_labels_batched = []
        sampled_rois_batched = []
        sampled_regression_targets_batched = []
        sampled_box_regression_batched = []
        sampled_class_logits_batched = []

        # gts for isr_p
        gts = []
        last_max_gt = 0

        for i in range(num_images):
            class_logits = class_logits_batched[i]
            box_regression = box_regression_batched[i]
            labels = labels_batched[i]
            rois = rois_batched[i]
            regression_targets = regression_targets_batched[i]

            sampled_pos_inds = torch.nonzero(labels > 0).squeeze(1)
            num_pos = sampled_pos_inds.size(0)

            neg_inds = torch.nonzero(labels == 0).squeeze(1)
            neg_rois = rois.index_select(0, neg_inds)
            neg_box_regression = box_regression.index_select(0, neg_inds)
            num_neg = neg_inds.size(0)
            if num_neg == 0:
                sampled_neg_inds = neg_inds
                neg_label_weights = []
            else:
                # PISA isr_n neg sampling
                with torch.no_grad():
                    neg_class_logits = class_logits.index_select(0, neg_inds)
                    # original_neg_class_loss = F.cross_entropy(neg_class_logits, neg_inds.new_full((num_neg, ), 81))
                    original_neg_class_loss = F.cross_entropy(neg_class_logits, neg_inds.new_full((num_neg,), 0), reduction="none")

                    max_score, argmax_score = neg_class_logits.softmax(-1)[:, :-1].max(-1)
                    valid_inds = (max_score > self.score_threshold).nonzero().view(-1)
                    invalid_inds = (max_score <= self.score_threshold).nonzero().view(-1)
                    num_valid = valid_inds.size(0)
                    num_invalid = invalid_inds.size(0)
                    num_neg_expected = self.batch_size_per_image - num_pos
                    num_neg_expected = min(num_neg, num_neg_expected)
                    num_hlr = min(num_valid, num_neg_expected)
                    num_rand = num_neg_expected - num_hlr
                    if num_valid > 0:
                        valid_rois = neg_rois[valid_inds]
                        valid_max_score = max_score[valid_inds]
                        valid_argmax_score = argmax_score[valid_inds]
                        valid_bbox_pred = neg_box_regression[valid_inds]

                        # valid_bbox_pred shape: [num_valid, #num_classes, 4]
                        valid_bbox_pred = valid_bbox_pred.view(valid_bbox_pred.size(0), -1, 4)
                        selected_bbox_pred = valid_bbox_pred[range(num_valid), valid_argmax_score]
                        pred_bboxes = self.box_coder.decode(selected_bbox_pred, valid_rois)
                        pred_bboxes_with_score = torch.cat([pred_bboxes, valid_max_score[:, None]], -1)
                        group = nms_match(pred_bboxes_with_score.float(), self.iou_threshold)

                        # imp: importance
                        imp = original_neg_class_loss.new_zeros(num_valid)
                        for g in group:
                            g_score = valid_max_score[g]
                            # g_score has already sorted
                            rank = g_score.new_tensor(range(g_score.size(0)))
                            imp[g] = num_valid - rank + g_score
                        _, imp_rank_inds = imp.sort(descending=True)
                        _, imp_rank = imp_rank_inds.sort()
                        hlr_inds = imp_rank_inds[:num_neg_expected]

                        if num_rand > 0:
                            rand_inds = torch.randperm(num_invalid)[:num_rand]
                            select_inds = torch.cat(
                                [valid_inds[hlr_inds], invalid_inds[rand_inds]])
                        else:
                            select_inds = valid_inds[hlr_inds]

                        neg_label_weights = original_neg_class_loss.new_ones(num_neg_expected)

                        up_bound = max(num_neg_expected, num_valid)
                        imp_weights = (up_bound -
                                       imp_rank[hlr_inds].float()) / up_bound
                        neg_label_weights[:num_hlr] = imp_weights
                        neg_label_weights[num_hlr:] = imp_weights.min()
                        neg_label_weights = (self.bias +
                                             (1 - self.bias) * neg_label_weights).pow(self.k)
                        ori_selected_loss = original_neg_class_loss[select_inds.to(neg_inds.device)]
                        new_loss = ori_selected_loss * neg_label_weights
                        norm_ratio = ori_selected_loss.sum() / new_loss.sum()
                        if torch.isnan(norm_ratio) or norm_ratio.eq(float('inf')):
                            norm_ratio = 1.0
                        neg_label_weights *= norm_ratio
                    else:
                        neg_label_weights = original_neg_class_loss.new_ones(num_neg_expected)
                        select_inds = torch.randperm(num_neg)[:num_neg_expected]

                    sampled_neg_inds = neg_inds[select_inds.to(neg_inds.device)]
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
            # sampled_pos_inds_batched.append(torch.nonzero(labels[sampled_inds]>0).view(-1))
            # sampled_inds_batched.append(sampled_inds)
            sampled_labels_batched.append(labels[sampled_inds])
            sampled_regression_targets_batched.append(regression_targets[sampled_inds])
            sampled_box_regression_batched.append(box_regression[sampled_inds])
            sampled_class_logits_batched.append(class_logits[sampled_inds])
            sampled_rois_batched.append(rois[sampled_inds])
            if num_neg == 0:
                label_weights = sampled_pos_inds.new_ones(num_pos)
            else:
                label_weights = torch.cat([sampled_pos_inds.new_ones(num_pos), neg_label_weights], dim=0)
            label_weights_batched.append(label_weights)
            box_weights = sampled_inds.new_zeros(sampled_inds.size(0), 4)
            box_weights[sampled_pos_inds] = 1.0
            box_weights_batched.append(box_weights)
            gt_i = proposals[i].get_field("matched_idxs")[sampled_pos_inds]
            gts.append(gt_i + last_max_gt)
            if len(gt_i) != 0:
                last_max_gt = gt_i.max() + 1

        return sampled_labels_batched, sampled_regression_targets_batched, \
               sampled_class_logits_batched, sampled_box_regression_batched, \
               sampled_rois_batched, label_weights_batched, box_weights_batched, gts, sampled_pos_inds_batched

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        # apply isr_n with batched inputs
        sampled_labels_batched, sampled_regression_targets_batched, sampled_class_logits_batched, \
        sampled_box_regression_batched, sampled_rois_batched, label_weights_batched, \
        box_weights_batched, gts, sampled_pos_inds_batched = self.isr_n(class_logits, box_regression)
        labels = torch.cat(sampled_labels_batched, dim=0)
        class_logits = cat(sampled_class_logits_batched, dim=0)
        box_regression = torch.cat(sampled_box_regression_batched, dim=0)
        regression_targets = torch.cat(sampled_regression_targets_batched, dim=0)
        rois = torch.cat(sampled_rois_batched, dim=0)
        label_weights = torch.cat(label_weights_batched, dim=0)
        box_weights = torch.cat(box_weights_batched, dim=0)
        device = class_logits.device

        pos_label_inds = torch.nonzero(labels > 0).squeeze(1)
        pos_labels = labels.index_select(0, pos_label_inds)
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * pos_labels[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        index_select_indices = ((pos_label_inds[:, None]) * box_regression.size(1) + map_inds).view(-1)
        pos_box_pred_delta = box_regression.view(-1).index_select(0, index_select_indices).view(map_inds.shape[0],
                                                                                                map_inds.shape[1])
        pos_box_target_delta = regression_targets.index_select(0, pos_label_inds)
        pos_rois = rois.index_select(0, pos_label_inds)

        if self.loss == "GIoULoss" and self.decode:
            # target is not encoded
            pos_box_target = pos_box_target_delta
        else:
            pos_box_target = self.box_coder.decode(pos_box_target_delta, pos_rois)
        pos_box_pred = self.box_coder.decode(pos_box_pred_delta, pos_rois)

        # Apply ISR-P
        # use default isr_p config: k=2 bias=0
        bbox_inputs = [labels, label_weights, regression_targets, box_weights, pos_box_pred, pos_box_target,
                       pos_label_inds, pos_labels]


        if self.use_isr_p:
            labels, label_weights, regression_targets, box_weights = isr_p(
                class_logits,
                bbox_inputs,
                torch.cat(gts, dim=0),
                self.cls_loss)

        if self.use_isr_n or self.use_isr_p:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            box_weights = box_weights.float()
            label_weights = label_weights.float()
        else:
            label_weights = None
            box_weights = None
            avg_factor = labels.size(0)

        classification_loss = self.cls_loss(class_logits,
                                            labels,
                                            weight=label_weights,
                                            avg_factor=avg_factor
                                            )

        if self.loss == "SmoothL1Loss":
            box_loss = smooth_l1_loss(
                pos_box_pred_delta.float(),
                pos_box_target_delta.float(),
                weight=box_weights,
                size_average=False,
                beta=1,
            )
            box_loss = box_loss / labels.numel()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            if self.carl:
                loss_carl = carl_loss(
                    class_logits,
                    pos_label_inds,
                    pos_labels,
                    pos_box_pred_delta,
                    pos_box_target_delta,
                    smooth_l1_loss,
                    k=1,
                    bias=0.2,
                    avg_factor=regression_targets.size(0))
            # end.record()
            # torch.cuda.synchronize()
            # print("carl loss time: ", start.elapsed_time(end))
        elif self.loss == "GIoULoss":
            if pos_box_pred.size()[0] > 0:
                if box_weights is not None:
                    box_weights = box_weights.index_select(0, pos_label_inds)
                box_loss = self.giou_loss(
                    pos_box_pred.float(),
                    pos_box_target.float(),
                    weight=box_weights,
                    avg_factor=labels.numel()
                )
            else:
                box_loss = box_regression.sum() * 0
            if self.carl:
                loss_carl = carl_loss(
                    class_logits,
                    pos_label_inds,
                    pos_labels,
                    pos_box_pred,
                    pos_box_target,
                    self.giou_loss_carl,
                    k=1,
                    bias=0.2,
                    avg_factor=regression_targets.size(0))

        if self.carl:
            return classification_loss, box_loss, loss_carl
        else:
            return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    # TODO: add pisa sampler
    fg_bg_sampler = PositiveSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = PISALossOnePassComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg,
        cfg.MODEL.ROI_BOX_HEAD.DECODE,
        cfg.MODEL.ROI_BOX_HEAD.LOSS,
        cfg.MODEL.ROI_BOX_HEAD.CARL,
        cfg.MODEL.ROI_BOX_HEAD.ISR_P,
        cfg.MODEL.ROI_BOX_HEAD.ISR_N,
        cfg.MODEL.ROI_BOX_HEAD.GIOU_BOX_WEIGHT,
        cfg.MODEL.ROI_BOX_HEAD.GIOU_CARL_WEIGHT,
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_HEADS.K,
        cfg.MODEL.ROI_HEADS.BIAS,
        cfg.MODEL.ROI_HEADS.SCORE_THRESHOLD,
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLD
    )

    return loss_evaluator
