import glob
import os
from PIL import Image
import cv2, numpy as np
import pickle
from src.utils.utils import cv2_to_pil, pil_to_cv2

class State:
    def __init__(self, conf):
        self.input_directory = conf.get('imagefolder')
        self.filetype = conf.get('imagetype')
        self.cropped_directory = conf.get('croppedfolder')
        self.splitting_directory = conf.get('splittedfolder')
        self.mask_directory = conf.get("maskfolder")
        self.pickle = conf.get('picklefolder')
        self.save_flag = conf.get('save_images')
        self.clean()

    def clean(self):
        filenames = list(os.listdir(self.input_directory))
        filenames = list(filter(lambda x: x.lower().endswith((self.filetype)), filenames))
        self.images = dict()
        for filename in filenames:
            image_name = os.path.basename(filename).split('.')[0]
            self.images[image_name] = {}
            image_path = os.path.join(self.input_directory,filename)
            image = Image.open(image_path)
            self.images[image_name]['original'] = image
            self.images[image_name]['masks'] = {}

    def make_overall_image(self, image_name, masks, channel):
        blended = None
        base_image = self.get_channel(image_name, channel)
        if len(masks) > 0:
            h, w = masks[0]['segmentation'].shape
            overlay = np.zeros((h, w, 3), dtype=np.uint8)  # Immagine vuota per le maschere
            for mask in masks:
                mask_img = mask['segmentation'].astype(np.uint8)  # Converti la maschera in uint8 (0-1 -> 0-255)
                color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Colore casuale
                overlay[mask_img > 0] = color  # Applica colore alla maschera
            if base_image is not None:
                base_img = pil_to_cv2(base_image)
                base_img = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)  # Converti in BGR se l'immagine di base è RGB
                alpha = 0.35
                blended = cv2.addWeighted(base_img, 1 - alpha, overlay, alpha, 0)
            else:
                blended = overlay
        return blended

    def remove(self, image_name: str):
        del self.images[image_name]

    def get_base_images(self):
        return list(self.images.keys())

    def get_original(self, image_name: str) -> Image:
        return self.images[image_name]['original']

    def set_original(self, image_name: str, image: Image):
        self.images[image_name]['original'] = image

    def get_channel(self, image_name: str, channel_name: str) -> Image:
        return self.images[image_name][channel_name]

    def get_masks(self, image_name: str) -> dict():
        return self.images[image_name]['masks']

    def add_original(self, image_name, image):
        self.images[image_name]['cropped'] = image
        filename = f"{image_name}_cropped.png"
        self.save_image_and_log(image,self.cropped_directory,filename) #todo: change function, this is a copy&paste of add_cropped

    def add_cropped(self, image_name, image):
        self.images[image_name]['cropped'] = image
        filename = f"{image_name}_cropped.png"
        self.save_image_and_log(image,self.cropped_directory,filename)

    def add_channel(self, image_name, image, channel):
        self.images[image_name][channel] = image
        filename = f"{image_name}_{channel}.png"
        self.save_image_and_log(image,self.splitting_directory,filename)

    def add_mask(self, image_name, mask, channel):
        channels = self.images[image_name]['masks'].keys()
        if not channel in channels:
            self.images[image_name]['masks'][channel] = {'singles': list(), 'merged': None}
        self.images[image_name]['masks'][channel]['singles'].append(mask)
        filename = f"{image_name}_{channel}_mask_{mask['id']}.png"
        mask_pillow = cv2_to_pil(mask['segmentation'])
        self.save_image_and_log(mask_pillow, self.mask_directory, filename)

    def add_masks(self, image_name, masks, channel):
        for mask in masks:
            self.add_mask(image_name, mask, channel)
        merged = self.make_overall_image(image_name, masks, channel)
        self.images[image_name]['masks'][channel]['merged'] = merged
        merged_pillow = cv2_to_pil(merged)
        filename = f"{image_name}_{channel}_mergedmasks.png"
        self.save_image_and_log(merged_pillow, self.mask_directory, filename)

    def save_image_and_log(self, image, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.save_flag:
            output_path = os.path.join(directory, filename)
            image.save(output_path)
            print(f"Immagine salvata: {output_path}")

    def save_pickle(self, image_name):
        filename = f'{image_name}.pickle'
        output_path = os.path.join(self.pickle,filename)
        with open(output_path, "wb") as f:
            pickle.dump(self.images[image_name], f)

    def check_pickle(self, image_name):
        filename = f'{image_name}.pickle'
        pattern = os.path.join(self.pickle, filename)
        check = glob.glob(pattern)
        return check

    def load_pickle(self):
        self.images = dict()
        filenames = list(os.listdir(self.pickle))
        for filename in filenames:
            image_name = os.path.splitext(filename)[0]
            input_path = os.path.join(self.pickle, filename)
            with open(input_path, "rb") as f:
                temp = pickle.load(f)
                self.images[image_name] = {**self.images, **temp}

