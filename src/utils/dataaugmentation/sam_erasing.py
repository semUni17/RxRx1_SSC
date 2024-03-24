from random import randint, sample

import numpy as np
import cv2

from src.methods.segmentation.sam.sam import SAM


class SAMErasing:
    ERASE = "erase"
    KEEP = "keep"

    def __init__(
            self,
            model: SAM,
            mask_mode=ERASE,
            scale=(0.2, 0.2, 0.6, 0.6),
            kernel_size=(10, 10),
            thresh=50,
            min_area=2000,
            percentage=0.5
    ):
        self.model = model
        self.mask_mode = mask_mode
        self.scale = scale
        self.kernel_size = kernel_size
        self.thresh = thresh
        self.min_area = min_area
        self.percentage = percentage

        self.initialize()

    def initialize(self):
        pass

    def __call__(self, x):
        if self.model.mask_mode == SAM.PREDICTOR:
            x = self.erase_predictor(x)
        elif self.model.mask_mode == SAM.GENERATOR:
            x = self.erase_generator(x)
        return x

    def erase_predictor(self, x):
        #cv2.imshow("original image", x)
        box = self.generate_bounding_box(x)
        masks = self.model.mask(x, box)
        mask = masks[0]
        mask = self.fill_expand_mask(mask)
        if self.mask_mode == self.ERASE:
            x[mask != 0] = 0
        elif self.mask_mode == self.KEEP:
            x[mask == 0] = 0
        #cv2.imshow("image", x)
        #cv2.waitKey(1)
        return x

    def generate_bounding_box(self, x):
        width, height = x.shape[0], x.shape[1]
        min_b_w, min_b_h = round(width*self.scale[0]), round(height*self.scale[1])
        max_b_w, max_b_h = round(width*self.scale[2]), round(height*self.scale[3])
        b_w, b_h = randint(min_b_w, max_b_w), randint(min_b_h, max_b_h)
        b_x_1, b_y_1 = randint(0, width-b_w), randint(0, height-b_h)
        b_x_2, b_y_2 = b_x_1+b_w, b_y_1+b_h
        box = np.array([b_x_1, b_y_1, b_x_2, b_y_2])
        return box

    def fill_expand_mask(self, mask):
        mask = mask.astype(np.uint8) * 255
        mask = cv2.blur(mask, ksize=self.kernel_size)
        _, mask = cv2.threshold(mask, thresh=self.thresh, maxval=255, type=cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=self.kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def erase_generator(self, x):
        #cv2.imshow("original image", x)
        masks = self.model.mask(x)
        masks = [m for m in masks if m["area"] <= self.min_area]
        n_masks = len(masks)
        n_random = round(n_masks*self.percentage)
        masks = sample(masks, n_random)
        if self.mask_mode == self.ERASE:
            for m in masks:
                x[m["segmentation"] != 0] = 0
        elif self.mask_mode == self.KEEP:
            mask = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
            for m in masks:
                mask[m["segmentation"] != 0] = 1
            mask = self.fill_expand_mask(mask)
            x[mask == 0] = 0
        #cv2.imshow("image", x)
        #cv2.waitKey(1)
        return x
