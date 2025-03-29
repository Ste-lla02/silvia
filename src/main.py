import sys, os

from src.core.core_model import State
from src.segmentation.evaluator import MaskFeaturing
from src.preprocessing.image_cropper import crop_image_with_polygon
from src.preprocessing.preprocessing import splitting_broker
from src.segmentation.sam_generator import Segmenter
from src.utils.configuration import Configuration
from src.utils.utils import FileCleaner


def build(conf: Configuration):
    # cleaning
    cleaner = FileCleaner()
    cleaner.clean()
    # Starting
    images = State(configuration)
    for image_filename in images.get_base_images():
        # Cropping
        image_name = os.path.basename(image_filename).split('.')[0]
        image = images.get_original(image_name)
        cropped_image = crop_image_with_polygon(image, image_name)
        images.add_cropped(image_name,cropped_image)
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
            masks = segmenter.mask_generation(to_segment, image_name, channel)
            masks = list(filter(lambda x: f.filter(x), masks))
            images.add_masks(image_name,masks,channel)
        images.save_pickle()
        #final_mask = Segmenter.mask_voting(all_mask)

def progress(conf: Configuration):
    pass


functions = {
    'build': build,
    'progress': progress
}

if __name__ == '__main__':
    configuration = Configuration(sys.argv[1])
    command = functions[sys.argv[2]]
    command(configuration)






