import numpy as np
import pandas as pd
import os


class Analysis:
    def __init__(self, conf):
        self.folder = conf.get('analysisfolder')
        self.channels = conf.get('analysis_channels')
        self.max_dist = conf.get('max_dist')
        self._next_c_id = 0


    def get_channels(self):
        return list(self.channels)

    def get_analysisfolder(self):
        return self.folder

    def get_max_dist(self):
        return self.max_dist

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
                'perimeter_px': perimeter,
                'inner_green': mask['inner_green'],
                'outer_green': mask['outer_green'],
            })
        return pd.DataFrame.from_records(records)

    def match_mask_ids(self, prev_df: pd.DataFrame, curr_df: pd.DataFrame, max_dist):
        curr = curr_df.copy()

        if prev_df is None or prev_df.empty:
            return curr

        prev_coords = prev_df[['centroid_x', 'centroid_y']].values
        prev_ids = prev_df['c_id'].values

        curr_coords = curr[['centroid_x', 'centroid_y']].values
        for i, (x, y) in enumerate(curr_coords):
            dists = np.hypot(prev_coords[:, 0] - x,
                             prev_coords[:, 1] - y)
            j = np.argmin(dists)
            if dists[j] <= max_dist:
                curr.at[i, 'c_id'] = int(prev_ids[j])
        return curr

    def add_date(self, curr_df: pd.DataFrame, image_filename):
        base = os.path.splitext(image_filename)[0]
        date_str = f"{base[:4]}-{base[4:6]}"
        curr_df['date'] = date_str
        return curr_df

