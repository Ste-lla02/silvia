import sys, os
from src.core.core_model import State
from src.segmentation.channel_voter import Voter
from src.segmentation.evaluator import MaskFeaturing
from src.preprocessing.image_cropper import crop_image_with_polygon
from src.preprocessing.preprocessing import splitting_broker
from src.segmentation.sam_generator import Segmenter
from src.utils.configuration import Configuration
from src.utils.utils import FileCleaner, send_ntfy_notification, send_ntfy_error

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
                    masks = segmenter.mask_generation(to_segment)
                    masks = list(filter(lambda x: f.filter(x), masks))
                    images.add_masks(image_name, masks, channel)
                image_masks = images.get_masks(image_name)
                voter = Voter(conf)
                final_masks = voter.mask_voting(image_masks)
                # Serializing
                images.save_pickle(image_name)
                print(f"Pickle salvato: {image_name}.pickle")
            except Exception as e:
                print(image_name, str(e))
                send_ntfy_error(topic, image_name, str(e))
            finally:
                images.remove(image_name)
    send_ntfy_notification(topic)

def progress(conf: Configuration):
    topic = conf.get('ntfy_topic')
    images = State(conf)
    try:
        images.load_pickle()
    except Exception as e:
        print(str(e))
        send_ntfy_error(topic, 'pkl', str(e))

def clean(conf: Configuration):
    cleaner = FileCleaner(conf)
    cleaner.clean()

functions = {
    'build': build,
    'clean': clean,
    'progress': progress
}

if __name__ == '__main__':
    configuration = Configuration(sys.argv[1])
    command = functions[sys.argv[2]]
    command(configuration)






