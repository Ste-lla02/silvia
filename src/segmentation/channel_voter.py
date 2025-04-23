import os
from typing import List

import cv2, numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from src.segmentation.evaluator import MaskFeaturing
from src.utils.configuration import Configuration
import gc

class Voter:
    def __init__(self, conf):
        self.priority_channels = conf.get('priority_channels')
        self.secondary_channels = conf.get('secondary_channels')
        pass

    def mask_voting(self, mask_list: dict):
        return mask_list[0]