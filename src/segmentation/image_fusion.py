from typing import List
import numpy as np
import itertools

from src.utils.configuration import Configuration


@staticmethod
def bbox_from_mask(mask: np.ndarray):
    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, 0, 0]

    x_min = int(np.min(x_indices))
    x_max = int(np.max(x_indices))
    y_min = int(np.min(y_indices))
    y_max = int(np.max(y_indices))

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return [x_min, y_min, width, height]

class Fusion:
    def __init__(self, conf):
        self.masters = conf.get('masters')
        self.slaves = conf.get('slaves')
        self.is_k = conf.get('inter_slave_operator')
        self.im_k = conf.get('inter_master_operator')
        self.fusion_channel = conf.get('fusion_channel')
        self.th1=conf.get('th1')
        self.th2=conf.get('th2')
        self._id_counter = itertools.count(0)

    def get_masters(self):
        return self.masters

    def get_slaves(self):
        return self.slaves

    def get_channels(self):
        retval = list(self.masters)
        retval.extend(self.slaves)
        return retval

    def get_fusion_channel(self):
        return self.fusion_channel

    def iou_overlap(self, mask1, mask2):
        seg1 = mask1['segmentation']
        seg2 = mask2['segmentation']
        retval = None
        intersection_sum = np.logical_and(seg1, seg2).sum()
        union_sum = np.logical_or(seg1, seg2).sum()
        if union_sum != 0:
            iou = intersection_sum / union_sum
            if iou > self.th1:
                if iou >= 0.5:
                    retval = self.merge_union(mask1, mask2)
                else:
                    retval = self.merge_intersecion(mask1, mask2)
                mask1['merged'] = True
                mask2['merged'] = True
        return retval

    def merge_union(self, mask1, mask2):
        union = {}
        seg = np.logical_or(mask1['segmentation'], mask2['segmentation'])
        union['segmentation'] = seg
        union['bbox'] = bbox_from_mask(seg)
        union['area'] = int(np.sum(seg))
        union['crop_box'] = mask1['crop_box']
        return union

    def merge_intersecion(self, mask1, mask2):
        intersection = {}
        seg = np.logical_and(mask1['segmentation'], mask2['segmentation'])
        intersection['segmentation'] = seg
        intersection['bbox'] = bbox_from_mask(seg)
        area = int(np.sum(seg))
        intersection['area'] = area
        intersection['crop_box'] = mask1['crop_box']
        return intersection

    def __fusion_all(self, channels_to_merge):
        """
        Fusion with ALL policy: returns only the masks that overlap across ALL
        the considered channels.
        """
        merged_masks = list()
        channel_names = list(channels_to_merge.keys())
        while len(channel_names) > 1:
            channel = channel_names.pop(0)
            masks = list(channels_to_merge[channel])
            masks = list(filter(lambda x: x['merged'] == False,masks))
            print(f"{channel} has {len(masks)} masks")
            for mask in masks:
                for other_channel in channel_names:
                    other_masks = list(channels_to_merge[other_channel])
                    for other_mask in other_masks:
                        m = Fusion.iou_overlap(self, mask, other_mask)
                        if m is not None:
                            merged_masks.append(m)
        return merged_masks

    def __fusion_any(self, channels_to_merge):
        """
        Fusion with ANY policy: returns the masks that are present in ALL the
        considered channels, even if they do not overlap, without duplicates.
        """
        merged_masks = list()
        channel_names = list(channels_to_merge.keys())
        while len(channel_names) > 1:
            channel = channel_names.pop(0)
            masks = list(channels_to_merge[channel])
            masks = list(filter(lambda x: x['merged'] == False,masks))
            merged_masks.extend(masks)
            other_masks = list()
            first = True
            for mask in masks:
                for other_channel in channel_names:
                    if first:
                        other_masks = list(channels_to_merge[other_channel])
                        first = False
                    for other_mask in other_masks:
                        other_flag = Fusion.iou_overlap(self, mask, other_mask)
                        other_mask['merged'] = other_flag
                    other_masks = list(filter(lambda x: x['merged'] == False, other_masks))
            merged_masks.extend(other_masks)
        return merged_masks

    def __fusion_policy(self, channels_to_merge, k):
        if not channels_to_merge:
            return []
        if k == 'any':
            return self.__fusion_any(channels_to_merge)
        if k == 'all':
            return self.__fusion_all(channels_to_merge)
        raise ValueError(f"Unknown fusion policy: {k}")

    def _ensure_id(self, masks: List[dict] | None):
        for mask in masks:
            if 'id' not in mask or mask['id'] is None:
                mask['id'] = str(next(self._id_counter))
        return masks

    def mask_voting(self, mask_list: dict, channels: List[str]):
        internal_slaves = {key: mask_list[key] for key in channels if key in self.slaves}
        slave_fusion  = self.__fusion_policy(internal_slaves, self.is_k)

        internal_masters = {key: mask_list[key] for key in channels if key in self.masters}
        internal_masters['slaves_merged'] = slave_fusion

        final = self.__fusion_policy(internal_masters, self.im_k)
        final = self._ensure_id(final)
        return final