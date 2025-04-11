import os
from typing import List

import cv2, numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from src.segmentation.evaluator import MaskFeaturing
from src.utils.configuration import Configuration
import gc

class Voter:
    def __init__(self):
        configuration = Configuration()
        pass

    def mask_voting(self, mask_list: dict):
        pass