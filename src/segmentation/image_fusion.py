import os
from typing import List, Tuple

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
        retval = list(self.masters)
        retval.extend(self.slaves)
        return retval

    def __decode_operators(self, temp: str, n: int) -> int:
        k = 0
        if temp == 'any':
            k = 1
        elif temp == 'all':
            k = n
        else:
            k = int(temp)
        return k

    @staticmethod
    def overlapping_masks(mask1: dict(), mask2: dict()) -> bool:
        #todo: improve the detection of the cropping box
        x1_min, y1_min, x1_max, y1_max = mask1['crop_box']
        x2_min, y2_min, x2_max, y2_max = mask2['crop_box']
        retval = not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
        return retval

    @staticmethod
    def merge_masks(mask1: dict(), mask2: dict()) -> dict():
        retval = dict(mask1)
        #todo: implement the method
        return retval

    def __fusion(self, channels_to_merge, k):
        merged_masks = list()
        channel_names = list(channels_to_merge.keys())
        while len(channel_names) > 0:
            channel = channel_names.pop(0)
            masks = list(channels_to_merge[channel])
            masks = list(filter(lambda x: x['merged'] == False,masks))
            for mask in masks:
                merged_mask = dict(mask)
                candidate_masks = list()
                found = False
                for other_channel in channel_names:
                    other_masks = list(channels_to_merge[other_channel])
                    for other_mask in other_masks:
                        other_flag = Fusion.overlapping_masks(mask, other_mask)
                        found = found or other_flag
                        other_mask['merged'] = other_flag
                mask['merged'] = found





            pass
        pass

    def mask_voting(self, mask_list: dict, channels: List[str]):
        internal_slaves = {key: mask_list[key] for key in channels if key in self.slaves}
        slave_masks = self.__fusion(internal_slaves, self.is_k)
        pass