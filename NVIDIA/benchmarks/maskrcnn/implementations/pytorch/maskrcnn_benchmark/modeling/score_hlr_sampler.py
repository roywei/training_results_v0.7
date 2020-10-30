# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
from torch.nn import functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList
from mmcv.ops import nms_match


class ScoreHLRSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction, k=0.5, bias=0, score_threshold=0.05, iou_threshold=0.5):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.score_threshold = score_threshold
        self.k = k
        self.bias = bias
        self.iou_threshold = iou_threshold

    def set_model(self, feature_extractor, predictor):
        self.feature_extractor = feature_extractor
        self.predictor = predictor

    def __call__(self, matched_idxs,  regression_targets, prop_boxes, image_sizes,
                 features, box_coder, is_rpn=0, objectness=None):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        num_images = len(matched_idxs)
        with torch.no_grad():
            if num_images == 1:
                pos_idx = []
                neg_idx = []
                neg_label_weights_batched = []
                matched_idxs = [matched_idxs.view(-1)]
                # there is actually only 1 iteration of this for loop, but keeping the loop for completeness
                for matched_idxs_per_image in matched_idxs:
                    if objectness is not None:
                        objectness = objectness.view(-1)
                        positive = torch.nonzero((matched_idxs_per_image >= 1) * (objectness > -1)).squeeze(1)
                        negative = torch.nonzero((matched_idxs_per_image == 0) * (objectness > -1)).squeeze(1)
                    else:
                        positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
                        negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

                    num_pos = int(self.batch_size_per_image * self.positive_fraction)
                    # protect against not enough positive examples
                    num_pos = min(positive.numel(), num_pos)
                    num_neg = self.batch_size_per_image - num_pos
                    # protect against not enough negative examples
                    num_neg = min(negative.numel(), num_neg)

                    # randomly select positive and negative examples
                    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                    pos_idx_per_image = positive.index_select(0, perm1)

                    # create binary mask from indices
                    pos_idx_per_image_mask = torch.zeros_like(
                        matched_idxs_per_image, dtype=torch.bool
                    )

                    pos_idx_per_image_mask.index_fill_(0, pos_idx_per_image, 1)

                    pos_idx.append(pos_idx_per_image_mask)
                    if num_neg == 0:
                        return pos_idx, neg_idx, []

                    # run forward with all negative boxes to get scores
                    prop_boxes = prop_boxes.view(-1, 4)
                    neg_bboxes = prop_boxes[negative]
                    regression_targets = regression_targets.view(-1, 4)
                    neg_proposals = []
                    for i in range(num_images):
                        box = BoxList(neg_bboxes, image_size=image_sizes[i])
                        box.add_field("matched_idxs", matched_idxs_per_image[negative])
                        box.add_field("regression_targets", regression_targets[negative])
                        box.add_field("labels", matched_idxs_per_image[negative])
                        neg_proposals.append(box)
                    x = self.feature_extractor(features, neg_proposals)
                    cls_score, box_regression = self.predictor(x)
                    classification_loss = F.cross_entropy(cls_score, negative.new_full((negative.size(0),), 81),
                                                          reduction="none")
                    max_score, argmax_score = cls_score.softmax(-1)[:, :-1].max(-1)
                    valid_inds = (max_score > self.score_threshold).nonzero().view(-1)
                    invalid_inds = (max_score <= self.score_threshold).nonzero().view(-1)
                    num_valid = valid_inds.size(0)
                    num_invalid = invalid_inds.size(0)
                    num_expected = num_neg
                    num_hlr = min(num_valid, num_expected)
                    num_rand = num_expected - num_hlr
                    if num_valid > 0:
                        valid_rois = neg_bboxes[valid_inds]
                        valid_max_score = max_score[valid_inds]
                        valid_argmax_score = argmax_score[valid_inds]
                        valid_bbox_pred = box_regression[valid_inds]

                        # valid_bbox_pred shape: [num_valid, #num_classes, 4]
                        valid_bbox_pred = valid_bbox_pred.view(
                            valid_bbox_pred.size(0), -1, 4)
                        selected_bbox_pred = valid_bbox_pred[range(num_valid),
                                                             valid_argmax_score]
                        pred_bboxes = box_coder.decode(
                            selected_bbox_pred, valid_rois)
                        pred_bboxes_with_score = torch.cat(
                            [pred_bboxes, valid_max_score[:, None]], -1)
                        group = nms_match(pred_bboxes_with_score.float(), self.iou_threshold)

                        # imp: importance
                        imp = cls_score.new_zeros(num_valid)
                        for g in group:
                            g_score = valid_max_score[g]
                            # g_score has already sorted
                            rank = g_score.new_tensor(range(g_score.size(0)))
                            imp[g] = num_valid - rank + g_score
                        _, imp_rank_inds = imp.sort(descending=True)
                        _, imp_rank = imp_rank_inds.sort()
                        hlr_inds = imp_rank_inds[:num_expected]

                        if num_rand > 0:
                            rand_inds = torch.randperm(num_invalid)[:num_rand]
                            select_inds = torch.cat(
                                [valid_inds[hlr_inds], invalid_inds[rand_inds]])
                        else:
                            select_inds = valid_inds[hlr_inds]

                        neg_label_weights = cls_score.new_ones(num_expected)

                        up_bound = max(num_expected, num_valid)
                        imp_weights = (up_bound -
                                       imp_rank[hlr_inds].float()) / up_bound
                        neg_label_weights[:num_hlr] = imp_weights
                        neg_label_weights[num_hlr:] = imp_weights.min()
                        neg_label_weights = (self.bias +
                                             (1 - self.bias) * neg_label_weights).pow(
                            self.k)
                        ori_selected_loss = classification_loss[select_inds]
                        new_loss = ori_selected_loss * neg_label_weights
                        norm_ratio = ori_selected_loss.sum() / new_loss.sum()
                        neg_label_weights *= norm_ratio
                    else:
                        neg_label_weights = cls_score.new_ones(num_expected)
                        select_inds = torch.randperm(num_neg)[:num_expected]
                    neg_idx_per_image = negative.index_select(0, select_inds.to(negative.device))
                    neg_idx_per_image_mask = torch.zeros_like(
                        matched_idxs_per_image, dtype=torch.bool
                    )
                    neg_idx_per_image_mask.index_fill_(0, neg_idx_per_image, 1)
                    neg_idx.append(neg_idx_per_image_mask)
                    neg_label_weights_batched.append(neg_label_weights)

                return pos_idx, neg_idx, neg_label_weights_batched

            else:
                pos_idx = []
                neg_idx = []
                #matched_idxs = [matched_idxs.view(-1)]
                batch_neg_label_weights = []
                # there is actually only 1 iteration of this for loop, but keeping the loop for completeness
                for i in range(num_images):
                    matched_idxs_per_image = matched_idxs[i]

                    # if objectness is not None:
                    #     objectness = objectness.view(-1)
                    #     positive = torch.nonzero((matched_idxs_per_image >= 1) * (objectness > -1)).squeeze(1)
                    #     negative = torch.nonzero((matched_idxs_per_image == 0) * (objectness > -1)).squeeze(1)
                    # else:
                    positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
                    negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

                    num_pos = int(self.batch_size_per_image * self.positive_fraction)
                    # protect against not enough positive examples
                    num_pos = min(positive.numel(), num_pos)
                    num_neg = self.batch_size_per_image - num_pos
                    # protect against not enough negative examples
                    num_neg = min(negative.numel(), num_neg)

                    # randomly select positive and negative examples
                    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                    pos_idx_per_image = positive.index_select(0, perm1)


                    # create binary mask from indices
                    pos_idx_per_image_mask = torch.zeros_like(
                        matched_idxs_per_image, dtype=torch.bool
                    )

                    pos_idx_per_image_mask.index_fill_(0, pos_idx_per_image, 1)

                    pos_idx.append(pos_idx_per_image_mask)

                    # run forward with all negative boxes to get scores
                    prop_box = prop_boxes[i].unsqueeze(1).view(-1, 4)
                    neg_box = prop_box[negative]
                    regression_target = regression_targets[i].unsqueeze(1).view(-1, 4)
                    neg_proposals = []
                    box = BoxList(neg_box, image_size=image_sizes[i])
                    box.add_field("matched_idxs", matched_idxs_per_image[negative])
                    box.add_field("regression_targets", regression_target[negative])
                    box.add_field("labels", matched_idxs_per_image[negative])
                    neg_proposals.append(box)
                    x = self.feature_extractor(features, neg_proposals)
                    cls_score, box_regression = self.predictor(x)
                    classification_loss = F.cross_entropy(cls_score, negative.new_full((negative.size(0),), 81), reduction="none")
                    max_score, argmax_score = cls_score.softmax(-1)[:, :-1].max(-1)
                    valid_inds = (max_score > self.score_threshold).nonzero().view(-1)
                    invalid_inds = (max_score <= self.score_threshold).nonzero().view(-1)
                    num_valid = valid_inds.size(0)
                    num_invalid = invalid_inds.size(0)
                    num_expected = num_neg
                    num_hlr = min(num_valid, num_expected)
                    num_rand = num_expected - num_hlr
                    if num_valid > 0:
                        valid_rois = neg_box[valid_inds]
                        valid_max_score = max_score[valid_inds]
                        valid_argmax_score = argmax_score[valid_inds]
                        valid_bbox_pred = box_regression[valid_inds]

                        # valid_bbox_pred shape: [num_valid, #num_classes, 4]
                        valid_bbox_pred = valid_bbox_pred.view(
                            valid_bbox_pred.size(0), -1, 4)
                        selected_bbox_pred = valid_bbox_pred[range(num_valid),
                                                             valid_argmax_score]
                        pred_bboxes = box_coder.decode(
                             selected_bbox_pred, valid_rois)
                        pred_bboxes_with_score = torch.cat(
                            [pred_bboxes, valid_max_score[:, None]], -1)
                        group = nms_match(pred_bboxes_with_score.float(), self.iou_threshold)

                        # imp: importance
                        imp = cls_score.new_zeros(num_valid)
                        for g in group:
                            g_score = valid_max_score[g]
                            # g_score has already sorted
                            rank = g_score.new_tensor(range(g_score.size(0)))
                            imp[g] = num_valid - rank + g_score
                        _, imp_rank_inds = imp.sort(descending=True)
                        _, imp_rank = imp_rank_inds.sort()
                        hlr_inds = imp_rank_inds[:num_expected]

                        if num_rand > 0:
                            rand_inds = torch.randperm(num_invalid)[:num_rand]
                            select_inds = torch.cat(
                                [valid_inds[hlr_inds], invalid_inds[rand_inds]])
                        else:
                            select_inds = valid_inds[hlr_inds]

                        neg_label_weights = cls_score.new_ones(num_expected)

                        up_bound = max(num_expected, num_valid)
                        imp_weights = (up_bound -
                                       imp_rank[hlr_inds].float()) / up_bound
                        neg_label_weights[:num_hlr] = imp_weights
                        neg_label_weights[num_hlr:] = imp_weights.min()
                        neg_label_weights = (self.bias +
                                             (1 - self.bias) * neg_label_weights).pow(
                            self.k)
                        ori_selected_loss = classification_loss[select_inds]
                        new_loss = ori_selected_loss * neg_label_weights
                        norm_ratio = ori_selected_loss.sum() / new_loss.sum()
                        neg_label_weights *= norm_ratio
                    else:
                        neg_label_weights = cls_score.new_ones(num_expected)
                        select_inds = torch.randperm(num_neg)[:num_expected]

                    neg_idx_per_image = negative.index_select(0, select_inds.to(negative.device))
                    neg_idx_per_image_mask = torch.zeros_like(
                        matched_idxs_per_image, dtype=torch.bool
                    )
                    neg_idx_per_image_mask.index_fill_(0, neg_idx_per_image, 1)
                    neg_idx.append(neg_idx_per_image_mask)
                    batch_neg_label_weights.append(neg_label_weights)
                return pos_idx, neg_idx, batch_neg_label_weights

            ## this implements a batched random subsampling using a tensor of random numbers and sorting
            # else:
            #     matched_idxs_cat = matched_idxs
            #     device = matched_idxs_cat.device
            #     pos_samples_mask = (matched_idxs_cat >= 1) * (objectness > -1)
            #     num_pos_samples = pos_samples_mask.sum(dim=1)
            #     num_pos_samples_cum = num_pos_samples.cumsum(dim=0)
            #     max_pos_samples = torch.max(num_pos_samples)
            #     consec = torch.arange(max_pos_samples, device=device).repeat(num_images, 1)
            #     mask_to_hide = consec >= num_pos_samples.view(num_images, 1)
            #     rand_nums_batched = torch.rand([num_images, max_pos_samples], device=device)
            #     rand_nums_batched.masked_fill_(mask_to_hide, 2)
            #     rand_perm = rand_nums_batched.argsort(dim=1)
            #     max_pos_allowed = int(self.batch_size_per_image * self.positive_fraction)
            #     num_pos_subsamples = num_pos_samples.clamp(max=max_pos_allowed)
            #     subsampling_mask = rand_perm < num_pos_subsamples.view(num_images, 1)
            #     if num_images > 1:
            #         consec[1:, :] = consec[1:, :] + num_pos_samples_cum[:-1].view(num_images - 1, 1)
            #     sampling_inds = consec.masked_select(subsampling_mask)
            #     pos_samples_inds = pos_samples_mask.view(-1).nonzero().squeeze(1)
            #     pos_subsampled_inds = pos_samples_inds[sampling_inds]
            #
            #     neg_samples_mask = (matched_idxs_cat == 0) * (objectness > -1)
            #     num_neg_samples = neg_samples_mask.sum(dim=1)
            #     num_neg_samples_cum = num_neg_samples.cumsum(dim=0)
            #     max_neg_samples = torch.max(num_neg_samples)
            #     consec = torch.arange(max_neg_samples, device=device)
            #     consec = consec.repeat(num_images, 1)
            #     mask_to_hide = consec >= num_neg_samples.view(num_images, 1)
            #     rand_nums_batched = torch.rand([num_images, max_neg_samples], device=device)
            #     rand_nums_batched.masked_fill_(mask_to_hide, 2)
            #     rand_perm = rand_nums_batched.argsort(dim=1)
            #     num_subsamples = torch.min(num_neg_samples, self.batch_size_per_image - num_pos_subsamples)
            #     subsampling_mask = rand_perm < num_subsamples.view(num_images, 1)
            #     if num_images > 1:
            #         consec[1:, :] = consec[1:, :] + num_neg_samples_cum[:-1].view(num_images - 1, 1)
            #     sampling_inds = consec.masked_select(subsampling_mask)
            #     neg_samples_inds = neg_samples_mask.view(-1).nonzero().squeeze(1)
            #     neg_subsampled_inds = neg_samples_inds[sampling_inds]
            #     return pos_subsampled_inds, neg_subsampled_inds, num_pos_subsamples, num_subsamples, []

