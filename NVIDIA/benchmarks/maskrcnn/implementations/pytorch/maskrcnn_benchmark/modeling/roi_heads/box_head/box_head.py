# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss_pisa_onepass import make_roi_box_loss_evaluator as make_pisa_loss_onepass
from .loss_pisa_box_head import make_roi_box_loss_evaluator

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        if cfg.MODEL.ROI_HEADS.PISA_ONEPASS:
            self.loss_evaluator = make_pisa_loss_onepass(cfg)
            self.one_pass = True
        else:
            self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
            self.loss_evaluator.set_model(self.feature_extractor, self.predictor)
            self.one_pass = False

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        print("*****************")
        print("BOX_HEAD time benchmarking starting:")
        start = torch.cuda.Event(enable_timing=True)
        end_subsample = torch.cuda.Event(enable_timing=True)
        end_feature_extractor = torch.cuda.Event(enable_timing=True)
        end_predictor = torch.cuda.Event(enable_timing=True)
        end_loss_evaluator = torch.cuda.Event(enable_timing=True)

        start.record()
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets, features)
        end_subsample.record()
        torch.cuda.synchronize()
        print("BOX_HEAD subsample time: ", start.elapsed_time(end_subsample))
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        end_feature_extractor.record()
        torch.cuda.synchronize()
        print("BOX_HEAD feature_extractor time: ", end_subsample.elapsed_time(end_feature_extractor))
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        end_predictor.record()
        torch.cuda.synchronize()
        print("BOX_HEAD predictor time: ", end_feature_extractor.elapsed_time(end_predictor))

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        results = self.loss_evaluator(
            [class_logits.float()], [box_regression.float()]
        )
        end_loss_evaluator.record()
        torch.cuda.synchronize()
        print("BOX_HEAD loss_evaluator time: ", end_predictor.elapsed_time(end_loss_evaluator))
        if self.one_pass:
            end_feature_extractor_2 = torch.cuda.Event(enable_timing=True)
            x = self.feature_extractor(features, self.loss_evaluator._proposals)
            end_feature_extractor_2.record()
            torch.cuda.synchronize()
            print("BOX_HEAD feature_extractor_2 time: ", end_loss_evaluator.elapsed_time(end_feature_extractor_2))

        if len(results) > 2:
            loss_dict = dict(loss_classifier=results[0], loss_box_reg=results[1], loss_carl=results[2])
        else:
            loss_dict = dict(loss_classifier=results[0], loss_box_reg=results[1])

        if self.one_pass:
            proposals = self.loss_evaluator._proposals

        print("*****************")
        return (
            x,
            proposals,
            loss_dict,
        )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
