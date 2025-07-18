import numpy as np
import pandas as pd
import os


class Analysis:
    def __init__(self, conf):
        self.folder = conf.get('analysisfolder')
        self.channels = conf.get('channels')
        self._next_c_id = 0
        pass

    def get_channels(self):
        return list(self.channels)

    def get_analysisfolder(self):
        return self.folder

    def save_analysisdata(self, folder_path, data, image_filename):
        file_name = f"{image_filename}_analysis.csv"
        fullpath = os.path.join(folder_path, file_name)
        data.to_csv(fullpath, index=False)
        print(f"Analisi di {image_filename} salvate in {fullpath}")

    def compute_centroid(self, mask_segmentation):
        coords = np.argwhere(mask_segmentation)
        return tuple(coords.mean(axis=0))

    def extract_mask_features(self, masks: list) -> pd.DataFrame:
        """
        Dà in input una lista di dizionari con chiave 'segmentation' e restituisce
        un DataFrame con le feature di ciascuna maschera e l'id temporale.
        """
        records = []
        for mask in masks:
            seg = mask['segmentation']
            cy, cx = self.compute_centroid(seg)
            perimeter = np.logical_xor(seg, np.roll(seg, 1, axis=0)).sum() + np.logical_xor(seg, np.roll(seg, 1,
                                                                                                         axis=1)).sum()
            # genera qui l'id univoco: channel_id
            c_id = self._next_c_id
            self._next_c_id += 1
            records.append({
                'c_id': c_id,
                'centroid_y': cy,
                'centroid_x': cx,
                'perimeter_px': perimeter
            })
        return pd.DataFrame.from_records(records)

    def match_mask_ids(self, prev_df: pd.DataFrame, curr_df: pd.DataFrame, max_dist: float = 1.0) -> pd.DataFrame:
        """
        Abbinamento greedy tra maschere di due DataFrame basato sulla distanza dei centroidi.
        Se la distanza minima è <= max_dist, eredita lo stesso ID, altrimenti ne assegna uno nuovo.
        """
        curr_df = curr_df.copy()
        curr_df['fc_id'] = -1
        next_id = prev_df['fc_id'].max() + 1 if 'fc_id' in prev_df else 0

        used_prev = set()
        for i, curr in curr_df.iterrows():
            dy = prev_df['centroid_y'].values - curr['centroid_y']
            dx = prev_df['centroid_x'].values - curr['centroid_x']
            dists = np.hypot(dy, dx)
            if len(dists) > 0:
                j = np.argmin(dists)
                if dists[j] <= max_dist and j not in used_prev:
                    curr_df.at[i, 'fc_id'] = prev_df.at[j, 'fc_id']
                    used_prev.add(j)
                else:
                    curr_df.at[i, 'fc_id'] = next_id
                    next_id += 1
            else:
                # nessun precedente, assegna nuovo ID
                curr_df.at[i, 'fc_id'] = next_id
                next_id += 1

        return curr_df
