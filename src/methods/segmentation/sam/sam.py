import os
import sys
from multipledispatch import dispatch

import numpy as np
import cv2

import torch
import torch.cuda
import torchvision

sys.path.append("segment-anything/")
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry


class SAM:
    PREDICTOR = "predictor"
    GENERATOR = "generator"

    def __init__(self, vit_type="vit_h", weights=None, mask_mode=GENERATOR, **kwargs):
        self.vit_type = vit_type
        self.weights = weights
        self.mask_mode = mask_mode
        self.kwargs = kwargs

        self.device = None
        self.model = None
        self.predictor = None
        self.generator = None

        self.initialize()

    def initialize(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = sam_model_registry[self.vit_type](checkpoint=self.weights)
        self.model.to(self.device)
        self.model.eval()

        if self.mask_mode == self.PREDICTOR:
            self.predictor = SamPredictor(self.model)
        elif self.mask_mode == self.GENERATOR:
            self.generator = SamAutomaticMaskGenerator(
                self.model,
                points_per_side=self.kwargs["points_per_side"],
                points_per_batch=self.kwargs["points_per_batch"],
                pred_iou_thresh=self.kwargs["pred_iou_thresh"],
                box_nms_thresh=self.kwargs["box_nms_thresh"],
                crop_nms_thresh=self.kwargs["crop_nms_thresh"],
                min_mask_region_area=self.kwargs["min_mask_region_area"]
            )

    @dispatch(np.ndarray, np.ndarray)
    def mask(self, image, prompt):
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            box=prompt
        )
        return masks

    @dispatch(np.ndarray)
    def mask(self, image):
        masks = self.generator.generate(image)
        return masks
