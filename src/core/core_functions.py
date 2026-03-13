import os
from src.core.core_model import State
from src.preprocessing.alternative_ndvi import inner_outer_green
from src.segmentation.image_fusion import Fusion
from src.segmentation.evaluator import MaskFeaturing
from src.preprocessing.image_cropper import crop_image_with_polygon
from src.preprocessing.preprocessing import splitting_broker
from src.segmentation.sam_generator import Segmenter
from src.spatial_analysis.spatial_analysis import Analysis
from src.utils.configuration import Configuration
from src.utils.utils import FileCleaner, send_ntfy_notification, send_ntfy_error
from itertools import chain

class Processor:
    def __init__(self, conf: Configuration, base_folder: str):
        self.conf = conf
        self.images = State(self.conf, base_folder)

    def full_process(self):
        self.build()
        self.fusion()
        self.analysis()

    def build(self):
        # Starting
        topic = self.conf.get('ntfy_topic')
        for image_filename in self.images.get_base_images():
            # Cropping
            image_name = os.path.basename(image_filename).split('.')[0]
            if not self.images.check_pickle(image_name):
                image = self.images.get_original(image_name) #todo: qui si genera l'errore
                try:
                    cropped_image = crop_image_with_polygon(image, self.conf)
                    self.images.add_cropped(image_name, cropped_image)
                    # Splitting
                    channels = self.conf.get('channels')
                    for channel in channels:
                        splitting_function = splitting_broker[channel]
                        splitted = splitting_function(cropped_image, image_name)
                        self.images.add_channel(image_name, splitted, channel)
                    # Segmentation
                    segmenter = Segmenter(self.conf)
                    f = MaskFeaturing(self.conf)
                    for channel in channels:
                        to_segment = self.images.get_channel(image_name, channel)
                        print(f"Masks Generation is running for image {image_name} channel {channel}...")
                        masks = segmenter.mask_generation(to_segment)
                        print(f"Filtering for image {image_name} channel {channel}...")
                        masks = list(filter(lambda x: f.filter(x), masks))
                        self.images.add_masks(image_name, masks, channel)
                        # Green Analysis
                        for mask in masks:
                            splitting_function = splitting_broker[channel]
                            splitted = splitting_function(cropped_image, image_name)
                            inner_outer_green(splitted, mask, channel)
                    # Serializing
                    self.images.save_pickle(image_name)
                except Exception as e:
                    print(image_name + ': ' +  str(e))
                    send_ntfy_error(topic, image_name, str(e))
                finally:
                    self.images.remove(image_name)
        send_ntfy_notification(topic)

    def fusion(self):
        topic = self.conf.get('ntfy_topic')
        self.images.load_pickle()
        fusion_engine = Fusion(self.conf)
        fusion_channel = fusion_engine.get_fusion_channel()
        channel_names = fusion_engine.get_channels()
        for image_filename in self.images.get_base_images():
            print(f"Fusion is running for image {image_filename}...")
            image = self.images.get_cropped(image_filename)
            masks = self.images.get_masks(image_filename, channel_names)
            self.images.clean_fusion(image_filename, fusion_channel)
            for mask in chain.from_iterable(masks.values()):
                mask['merged'] = False
            merged_masks = fusion_engine.mask_voting(masks, channel_names)
            print(f"Number of merged masks {len(merged_masks) if merged_masks else 0}")
            self.images.add_fusion(merged_masks, image, image_filename, fusion_channel)
            # Serializing
            self.images.save_fusion_pickle(image_filename)
            self.images.save_pickle(image_filename)
        send_ntfy_notification(topic)

    def analysis(self):
        topic = self.conf.get('ntfy_topic')
        self.images.load_pickle()
        analysis_engine = Analysis(self.conf)
        channel_names = analysis_engine.get_channels()
        folder_path = analysis_engine.get_analysisfolder()
        max_dist = analysis_engine.get_max_dist()
        prev_dfs = {ch: None for ch in channel_names}
        for image_filename in self.images.get_base_images():
            masks = self.images.get_masks(image_filename, channel_names)
            for channel_name in channel_names:
                print(f"Analysis is running for image {image_filename} and channel {channel_name}...")
                height, width, _ = self.images.get_image_shape(image_filename)
                curr_df = analysis_engine.extract_mask_features(height, width, masks[channel_name])
                curr_df = analysis_engine.add_date(curr_df, image_filename)
                if prev_dfs[channel_name] is not None:
                    curr_df = analysis_engine.match_mask_ids(prev_dfs[channel_name], curr_df, max_dist)
                else:
                    curr_df['c_id'] = curr_df['c_id']
                # Serializing
                analysis_engine.save_analysisdata(folder_path, curr_df, image_filename)
                prev_dfs[channel_name] = curr_df
            # Serializing
            # images.save_pickle(image_filename)
        send_ntfy_notification(topic)


# todo: aggiungere al processor
def clean(conf: Configuration):
    cleaner = FileCleaner(conf)
    cleaner.clean()
