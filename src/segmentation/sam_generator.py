import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from src.segmentation.evaluator import MaskFeaturing
from src.utils.configuration import Configuration
import gc
from src.utils.utils import pil_to_cv2, wget_download
import os

class Segmenter:
    def __init__(self):
        configuration = Configuration()
        # sam model file and parameters
        model_path = configuration.get('sam_model')
        sam_platform = configuration.get('sam_platform')
        sam_kind = configuration.get('sam_kind')
        self.model_downloader(model_path, sam_kind)
        sam = sam_model_registry[sam_kind](checkpoint=model_path)
        sam = sam.to(sam_platform)
        # Getting mask quality parameter values
        points_per_side = configuration.get('points_per_side')
        min_mask_quality = configuration.get('min_mask_quality')
        min_mask_stability = configuration.get('min_mask_stability')
        layers = configuration.get('layers')
        crop_n_points_downscale_factor = configuration.get('crop_n_points_downscale_factor')
        min_mask_region_area = configuration.get('min_mask_region_area')
        # generation of the segmenter
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=min_mask_quality,
            stability_score_thresh=min_mask_stability,
            crop_n_layers=layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area
        )

    def mask_generation(self, image):
        retval = list()
        gc.collect()
        cv2_image = pil_to_cv2(image)
        colored_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(colored_image)
        f = MaskFeaturing()
        for i, mask in enumerate(masks):
            id = {'id': i}
            properties = f.evaluation(mask)
            properties = {**properties, **id}
            retval.append({**mask, **properties})
        return retval

    def model_downloader(self, model_directory, sam_kind):
        model_directory = os.path.dirname(model_directory)
        # Mappa dei modelli
        sam_models = {
            'vit_h': "sam_vit_h_4b8939.pth",
            'vit_b': "sam_vit_b_01ec64.pth",
            'vit_l': "sam_vit_l_0b3195.pth"
        }

        if sam_kind not in sam_models:
            raise ValueError(f"SAM model not correct: {sam_kind}")

        # Costruzione dell'URL e del path
        model_name = sam_models[sam_kind]
        sam_url = f"https://dl.fbaipublicfiles.com/segment_anything/{model_name}"
        model_path = os.path.join(model_directory, model_name)

        # Creazione cartella se non esiste
        os.makedirs(model_directory, exist_ok=True)

        # Verifica presenza del file modello
        if not os.path.isfile(model_path):
            print(f"Downloading {model_name} to {model_path} ...")
            if wget_download(sam_url, model_path):
                print(f"Download {model_name} to {model_path} complete.")
            else:
                raise ValueError(f"Download {model_name} to {model_path} failed.")
        else:
            print(f"Model already downloaded: {model_path}")

        return model_path