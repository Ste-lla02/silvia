import os
from PIL import Image
import cv2, numpy as np


class State:
    def __init__(self, conf):
        self.input_directory = conf.get('imagefolder')
        self.filetype = conf.get('imagetype')
        self.cropped_directory = conf.get('croppedfolder')
        self.splitting_directory = conf.get('splittedfolder')
        self.mask_directory = conf.get("maskfolder")
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
                base_img = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)  # Converti in BGR se l'immagine di base è RGB
                alpha = 0.35
                blended = cv2.addWeighted(base_img, 1 - alpha, overlay, alpha, 0)
            else:
                blended = overlay
        return blended

    def get_base_images(self):
        return list(self.images.keys())

    def get_original(self, image_name: str) -> Image:
        return self.images[image_name]['original']

    def get_channel(self, image_name: str, channel_name: str) -> Image:
        return self.images[image_name][channel_name]

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
            self.images[image_name]['masks'][channel] = list()
        self.images[image_name]['masks'][channel].append(mask)
        filename = f"{image_name}_{channel}_mask_{mask['id']}.png"
        self.save_image_and_log(mask, self.mask_directory, filename)

    def add_masks(self, image_name, masks, channel):
        for mask in masks:
            self.add_mask(image_name, mask, channel)
        filename = f"{image_name}_{channel}_mergedmasks.png"
        merged = self.make_overall_image(masks, image_name, filename, channel)
        #todo: aggiungere il merged al dizionario
        self.save_image_and_log(merged, self.mask_directory, filename)

    def save_image_and_log(self, image, directory, filename):
        output_path = os.path.join(directory,filename)
        image.save(output_path)
        print(f"Immagine salvata: {output_path}")





