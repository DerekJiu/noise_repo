import torch.distributed as dist
import numpy as np
import ignite
import torch

from . import segmentation
from . import classification


class BinaryClassificationEvaluator(
        ignite.metrics.EpochMetric):
    def __init__(self, threshold=None, check_compute_fn=True):
        self._threshold = threshold
        super().__init__(self.compute_fn)

    def compute_fn(self, prob, y):
        return classification.get_binary_metrics(
            prob, y, threshold=self._threshold)

    def evaluate(self):
        return self.compute()
    
class SegmentationEvaluator(ignite.metrics.EpochMetric):
    def __init__(self, threshold=None, check_compute_fn=True):
        self._threshold = threshold
        super().__init__(self.compute_fn)

    def compute_fn(self, pred_masks, true_masks):

        return segmentation.get_segmentation_metrics(
            pred_masks, true_masks, threshold=self._threshold
        )

    def evaluate(self):
        return self.compute()
