import sys, os
from src.core.core_model import State
from src.segmentation.image_fusion import Fusion
from src.segmentation.evaluator import MaskFeaturing
from src.preprocessing.image_cropper import crop_image_with_polygon
from src.preprocessing.preprocessing import splitting_broker
from src.segmentation.sam_generator import Segmenter
from src.spatial_analysis.spatial_analysis import Analysis
from src.utils.configuration import Configuration
from src.utils.utils import FileCleaner, send_ntfy_notification, send_ntfy_error
from itertools import chain

def build(conf: Configuration):
    # Starting
    images = State(configuration)
    topic = conf.get('ntfy_topic')
    for image_filename in images.get_base_images():
        # Cropping
        image_name = os.path.basename(image_filename).split('.')[0]
        if not images.check_pickle(image_name):
            image = images.get_original(image_name)
            try:
                cropped_image = crop_image_with_polygon(image, image_name)
                images.add_cropped(image_name, cropped_image)
                # Splitting
                channels = configuration.get('channels')
                for channel in channels:
                    splitting_function = splitting_broker[channel]
                    splitted = splitting_function(cropped_image, image_name)
                    images.add_channel(image_name, splitted, channel)
                # Segmentation
                segmenter = Segmenter()
                f = MaskFeaturing()
                for channel in channels:
                    to_segment = images.get_channel(image_name, channel)
                    print(f"Masks Generation is running for channel {channel}...")
                    masks = segmenter.mask_generation(to_segment)
                    print(f"Filtering for channel {channel}...")
                    masks = list(filter(lambda x: f.filter(x), masks))
                    images.add_masks(image_name, masks, channel)
                # Serializing
                images.save_pickle(image_name)
            except Exception as e:
                send_ntfy_error(topic, image_name, str(e))
            finally:
                images.remove(image_name)
    send_ntfy_notification(topic)

def fusion(conf: Configuration):
    topic = conf.get('ntfy_topic')
    images = State(conf)
    images.load_pickle()
    fusion_engine = Fusion(conf)
    fusion_channel = fusion_engine.get_fusion_channel()
    channel_names = fusion_engine.get_channels()
    for image_filename in images.get_base_images():
        print(f"Fusion is running for image {image_filename}...")
        image = images.get_cropped(image_filename)
        masks = images.get_masks(image_filename, channel_names)
        for mask in chain.from_iterable(masks.values()):
            mask['merged'] = False
        merged_masks = fusion_engine.mask_voting(masks, channel_names)
        images.add_fusion(merged_masks, image, image_filename, fusion_channel)
        # Serializing
        images.save_fusion_pickle(image_filename)
        images.save_pickle(image_filename)
    send_ntfy_notification(topic)

def analysis(conf: Configuration):
    topic = conf.get('ntfy_topic')
    images = State(conf)
    images.load_pickle()
    analysis_engine = Analysis(conf)
    channel_name = analysis_engine.get_channel()
    for image_filename in images.get_base_images():
        print(f"Analysis is running for image {image_filename}...")
        image = images.get_channel(image_filename, channel_name)
        masks = images.get_fusion_masks(image_filename)


    send_ntfy_notification(topic)
    #todo implemetare
    pass

def clean(conf: Configuration):
    cleaner = FileCleaner(conf)
    cleaner.clean()

functions = {
    'build': build,
    'clean': clean,
    'fusion': fusion,
    'analysis': analysis
}

if __name__ == '__main__':
    configuration = Configuration(sys.argv[1])
    command = functions[sys.argv[2]]
    command(configuration)






