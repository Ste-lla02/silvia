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
        self.is_k = conf.get('inter_slave_operator')
        self.im_k = conf.get('inter_master_operator')
        #temp = conf.get('inter_slave_operator')
        #self.is_k = self.__decode_operators(temp, len(self.slaves))
        #temp = conf.get('inter_master_operator')
        #self.im_k = self.__decode_operators(temp, len(self.masters))

    def get_masters(self):
        return self.masters

    def get_slaves(self):
        return self.slaves

    def get_channels(self):
        retval = list(self.masters)
        retval.extend(self.slaves)
        return retval

    def __decode_operators(self, temp: str, n: int) -> int:
        if temp == 'any':
            return 1
        elif temp == 'all':
            return n
        else:
            return int(temp)

    @staticmethod
    def overlapping_masks(mask1: dict(), mask2: dict()) -> bool:
        #todo: improve the detection of the cropping box
        x1_min, y1_min, x1_max, y1_max = mask1['bbox']
        x2_min, y2_min, x2_max, y2_max = mask2['bbox']
        retval = not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
        return retval

    @staticmethod
    def iou_overlap(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union != 0:
            iou = intersection / union
            if iou <= 0.5:
                return False
            else:
                return True

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
                        if found:
                            other_mask['merged'] = found
                            mask['merged'] = found
                            candidate_masks.append(mask)
                        else:
                            continue
            pass
        pass

    def __fusion_all(self, channels_to_merge):
        """
        Fusione con policy ALL: restituisce solo le maschere che
        si sovrappongono in TUTTI i canali considerati.
        """
        merged_masks = list()
        channel_names = list(channels_to_merge.keys())
        while len(channel_names) > 0:
            channel = channel_names.pop(0)
            masks = list(channels_to_merge[channel])
            masks = list(filter(lambda x: x['merged'] == False,masks))
            for mask in masks:
                merged_mask = dict(mask)
                found = False
                for other_channel in channel_names:
                    other_masks = list(channels_to_merge[other_channel])
                    for other_mask in other_masks:
                        other_flag = Fusion.iou_overlap(mask['segmentation'], other_mask['segmentation'])
                        found = found or other_flag
                        if found:
                            other_mask['merged'] = found
                            mask['merged'] = found
                            merged_masks.append(mask)
                            found = False
                        else:
                            continue
        return merged_masks

    def __fusion_any(self, channels_to_merge):
        """
        Fusione con policy ANY: restituisce le maschere che
        presenti in TUTTI i canali considerati, anche se non si sovrappongono.
        """
        merged_masks = list()
        channel_names = list(channels_to_merge.keys())
        while len(channel_names) > 0:
            channel = channel_names.pop(0)
            masks = list(channels_to_merge[channel])
            masks = list(filter(lambda x: x['merged'] == False,masks))
            for mask in masks:
                saw = True
                found = False
                for other_channel in channel_names:
                    other_masks = list(channels_to_merge[other_channel])
                    for other_mask in other_masks:
                        other_flag = Fusion.iou_overlap(mask['segmentation'], other_mask['segmentation'])
                        found = found or other_flag
                        if found:
                            other_mask['merged'] = found
                            mask['merged'] = found
                            merged_masks.append(mask)
                            found = False
                        elif not found:
                            other_mask['merged'] = saw
                            mask['merged'] = found
                            merged_masks.append(other_mask)
        return merged_masks

    def __fusion_policy(self, channels_to_merge, k):
        try:
            if k == 'any':
                return self.__fusion_any(channels_to_merge)
            elif k == 'all':
                return self.__fusion_all(channels_to_merge)
        except Exception as e:
            print(e)




    def mask_voting(self, mask_list: dict, channels: List[str]):
        internal_slaves = {key: mask_list[key] for key in channels if key in self.slaves}
        slave_fusion  = self.__fusion_policy(internal_slaves, self.is_k)

        internal_masters = {key: mask_list[key] for key in channels if key in self.masters}
        internal_masters['slaves_merged'] = slave_fusion

        final = self.__fusion_policy(internal_masters, self.im_k)
        return final