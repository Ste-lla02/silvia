from typing import List
import numpy as np

class Fusion:
    def __init__(self, conf):
        self.masters = conf.get('masters')
        self.slaves = conf.get('slaves')
        self.is_k = conf.get('inter_slave_operator')
        self.im_k = conf.get('inter_master_operator')
        self.fusion_channel = conf.get('fusion_channel')

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

    def __fusion_all(self, channels_to_merge):
        """
        Fusione con policy ALL: restituisce solo le maschere che
        si sovrappongono in TUTTI i canali considerati.
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
                        other_flag = Fusion.iou_overlap(mask['segmentation'], other_mask['segmentation'])
                        if other_flag:
                            merged_masks.append(mask)
                        else:
                            continue
        return merged_masks

    def __fusion_any(self, channels_to_merge):
        """
        Fusione con policy ANY: restituisce le maschere che
        presenti in TUTTI i canali considerati, anche se non si sovrappongono, senza duplicati.
        """
        merged_masks = list()
        channel_names = list(channels_to_merge.keys())
        while len(channel_names) > 1:
            channel = channel_names.pop(0)
            masks = list(channels_to_merge[channel])
            masks = list(filter(lambda x: x['merged'] == False,masks))
            merged_masks.extend(masks)
            #print(f"Numero di maschere del channel {channel} : {len(masks)}\n")
            #print(f"Numero di maschere in merged : {len(merged_masks)}\n")
            other_masks = list()
            first = True
            for mask in masks:
                for other_channel in channel_names:
                    if first:
                        other_masks = list(channels_to_merge[other_channel])
                        #print(f"First: numero di maschere del channel {other_channel} : {len(other_masks)}\n")
                        first = False
                    for other_mask in other_masks:
                        other_flag = Fusion.iou_overlap(mask['segmentation'], other_mask['segmentation'])
                        other_mask['merged'] = other_flag
                    other_masks = list(filter(lambda x: x['merged'] == False, other_masks))
            #print(f"Numero di maschere del secondo canale : {len(other_masks)}\n")
            merged_masks.extend(other_masks)
        #print(f"Final: Numero di maschere in merged: {len(merged_masks)}\n")
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