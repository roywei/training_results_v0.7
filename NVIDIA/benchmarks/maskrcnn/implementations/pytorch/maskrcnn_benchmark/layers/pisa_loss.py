import torch

def isr_p(cls_score,
          bbox_pred,
          bbox_targets,
          rois,
          pos_assigned_gt_inds,
          loss_cls,
          bbox_coder,
          k=2,
          bias=0,
          num_class=80):
    """Importance-based Sample Reweighting (ISR_P), positive part.

    Args:
        cls_score (Tensor): Predicted classification scores.
        bbox_pred (Tensor): Predicted bbox deltas.
        bbox_targets (tuple[Tensor]): A tuple of bbox targets, the are
            labels, label_weights, bbox_targets, bbox_weights, respectively.
        rois (Tensor): Anchors (single_stage) in shape (n, 4) or RoIs
            (two_stage) in shape (n, 5).
        pos_assigned_gt_inds (tensor): Sampling results.
        loss_cls (func): Classification loss func of the head.
        bbox_coder (obj): BBox coder of the head.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        num_class (int): Number of classes, default: 80.

    Return:
        tuple([Tensor]): labels, imp_based_label_weights, bbox_targets,
            bbox_target_weights
    """

    labels, label_weights, bbox_targets, bbox_weights = bbox_targets
    pos_label_inds = (labels > 0).nonzero().reshape(-1)
    pos_labels = labels[pos_label_inds]

    # if no positive samples, return the original targets
    num_pos = float(pos_label_inds.size(0))
    if num_pos == 0:
        return labels, label_weights, bbox_targets, bbox_weights

    # merge pos_assigned_gt_inds of per image to a single tensor
    gts = list()
    last_max_gt = 0
    for i in range(len(pos_assigned_gt_inds)):
        gt_i = pos_assigned_gt_inds[i]
        gts.append(gt_i + last_max_gt)
        if len(gt_i) != 0:
            last_max_gt = gt_i.max() + 1
    gts = torch.cat(gts)
    assert len(gts) == num_pos

    cls_score = cls_score.detach()
    bbox_pred = bbox_pred.detach()

    # For single stage detectors, rois here indicate anchors, in shape (N, 4)
    # For two stage detectors, rois are in shape (N, 5)
    if rois.size(-1) == 5:
        pos_rois = rois[pos_label_inds][:, 1:]
    else:
        pos_rois = rois[pos_label_inds]

    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_delta_pred = bbox_pred[pos_label_inds, pos_labels].view(-1, 4)
    else:
        pos_delta_pred = bbox_pred[pos_label_inds].view(-1, 4)

    # compute iou of the predicted bbox and the corresponding GT
    pos_delta_target = bbox_targets[pos_label_inds].view(-1, 4)
    pos_bbox_pred = bbox_coder.decode(pos_rois, pos_delta_pred)
    target_bbox_pred = bbox_coder.decode(pos_rois, pos_delta_target)
    ious = bbox_overlaps(pos_bbox_pred, target_bbox_pred, is_aligned=True)

    pos_imp_weights = label_weights[pos_label_inds]
    # Two steps to compute IoU-HLR. Samples are first sorted by IoU locally,
    # then sorted again within the same-rank group
    max_l_num = pos_labels.bincount().max()
    for label in pos_labels.unique():
        l_inds = (pos_labels == label).nonzero().view(-1)
        l_gts = gts[l_inds]
        for t in l_gts.unique():
            t_inds = l_inds[l_gts == t]
            t_ious = ious[t_inds]
            _, t_iou_rank_idx = t_ious.sort(descending=True)
            _, t_iou_rank = t_iou_rank_idx.sort()
            ious[t_inds] += max_l_num - t_iou_rank.float()
        l_ious = ious[l_inds]
        _, l_iou_rank_idx = l_ious.sort(descending=True)
        _, l_iou_rank = l_iou_rank_idx.sort()  # IoU-HLR
        # linearly map HLR to label weights
        pos_imp_weights[l_inds] *= (max_l_num - l_iou_rank).float() / max_l_num

    pos_imp_weights = (bias + pos_imp_weights * (1 - bias)).pow(k)

    # normalize to make the new weighted loss value equal to the original loss
    pos_loss_cls = loss_cls(
        cls_score[pos_label_inds], pos_labels, reduction_override='none')
    if pos_loss_cls.dim() > 1:
        ori_pos_loss_cls = pos_loss_cls * label_weights[pos_label_inds][:,
                                                                        None]
        new_pos_loss_cls = pos_loss_cls * pos_imp_weights[:, None]
    else:
        ori_pos_loss_cls = pos_loss_cls * label_weights[pos_label_inds]
        new_pos_loss_cls = pos_loss_cls * pos_imp_weights
    pos_loss_cls_ratio = ori_pos_loss_cls.sum() / new_pos_loss_cls.sum()
    pos_imp_weights = pos_imp_weights * pos_loss_cls_ratio
    label_weights[pos_label_inds] = pos_imp_weights

    bbox_targets = labels, label_weights, bbox_targets, bbox_weights
    return bbox_targets


def carl_loss(cls_score,
              labels,
              bbox_pred,
              bbox_targets,
              loss_bbox,
              k=1,
              bias=0.2,
              avg_factor=None,
              sigmoid=False,
              num_class=80):
    """Classification-Aware Regression Loss (CARL).

    Args:
        cls_score (Tensor): Predicted classification scores.
        labels (Tensor): Targets of classification.
        bbox_pred (Tensor): Predicted bbox deltas.
        bbox_targets (Tensor): Target of bbox regression.
        loss_bbox (func): Regression loss func of the head.
        bbox_coder (obj): BBox coder of the head.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        avg_factor (int): Average factor used in regression loss.
        sigmoid (bool): Activation of the classification score.
        num_class (int): Number of classes, default: 80.

    Return:
        dict: CARL loss dict.
    """
    pos_label_inds = (labels > 0).nonzero().reshape(-1)
    if pos_label_inds.numel() == 0:
        return dict(loss_carl=cls_score.sum()[None] * 0.)
    pos_labels = labels[pos_label_inds]

    # multiply pos_cls_score with the corresponding bbox weight
    # and remain gradient
    if sigmoid:
        pos_cls_score = cls_score.sigmoid()[pos_label_inds, pos_labels]
    else:
        pos_cls_score = cls_score.softmax(-1)[pos_label_inds, pos_labels]
    carl_loss_weights = (bias + (1 - bias) * pos_cls_score).pow(k)

    # normalize carl_loss_weight to make its sum equal to num positive
    num_pos = float(pos_cls_score.size(0))
    weight_ratio = num_pos / carl_loss_weights.sum()
    carl_loss_weights *= weight_ratio

    if avg_factor is None:
        avg_factor = bbox_targets.size(0)
    # if is class agnostic, bbox pred is in shape (N, 4)
    # otherwise, bbox pred is in shape (N, #classes, 4)
    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_bbox_preds = bbox_pred[pos_label_inds, pos_labels]
    else:
        pos_bbox_preds = bbox_pred[pos_label_inds]
    ori_loss_reg = loss_bbox(
        pos_bbox_preds,
        bbox_targets[pos_label_inds],
        reduction_override='none') / avg_factor
    loss_carl = (ori_loss_reg * carl_loss_weights[:, None]).sum()
    return dict(loss_carl=loss_carl[None])


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, 1))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])

        if mode == 'iou':
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
                bboxes2[..., 3] - bboxes2[..., 1])
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])

        if mode == 'iou':
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
                bboxes2[..., 3] - bboxes2[..., 1])
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    return ious

def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois
