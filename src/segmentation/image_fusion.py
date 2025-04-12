import os
from typing import List

import cv2, numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from src.segmentation.evaluator import MaskFeaturing
from src.utils.configuration import Configuration
import gc

class Fusion:
    def __init__(self, conf):
        self.masters = conf.get('masters')
        self.slaves = conf.get('slaves')
        temp = conf.get('inter_slave_operator')
        self.is_k = self.__decode_operators(temp, len(self.slaves))
        temp = conf.get('inter_master_operator')
        self.im_k = self.__decode_operators(temp, len(self.masters))

    def get_masters(self):
        return self.masters

    def get_slaves(self):
        return self.slaves

    def get_channels(self):
        retval = list(map(lambda x: ('master',x), self.masters))
        slaves = list(map(lambda x: ('slave',x), self.slaves))
        retval.extend(slaves)
        return retval


    def __decode_operators(self, temp, n):
        k = 0
        if temp == 'any':
            k = 1
        elif temp == 'all':
            k = n
        else:
            k = int(temp)
        return k

    def mask_voting(self, mask_list: dict):
        pass