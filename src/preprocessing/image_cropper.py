from PIL import Image, ImageDraw
from src.utils.configuration import Configuration
from src.utils.utils import adjust_coordinate_rectangle


# Definisci la funzione per ritagliare un'immagine in base ai punti del poligono
def crop_image_with_polygon(image, image_name):
    configuration = Configuration()
    points = configuration.get('areaofinterest_image')
    scaling_factor = configuration.get("image_scaling")
    # Carica l'immagine
    # Creare una maschera con lo stesso formato e dimensione dell'immagine
    mask = Image.new("L", image.size, 0)  # 'L' significa scala di grigi (0-255)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255)  # Disegna il poligono e riempilo con bianco (255)
    # Applica la maschera all'immagine originale
    result = Image.new("RGBA", image.size)  # Creare un'immagine con canale alpha
    result.paste(image, mask=mask)  # Applica la maschera per conservare solo il poligono
    # Ritaglia l'immagine al bounding box del poligono per risparmiare spazio
    x_min, x_max, y_min, y_max = adjust_coordinate_rectangle(points)
    result_cropped = result.crop((x_min, y_min, x_max, y_max))
    # Resizing
    new_size = (int(result_cropped.width * scaling_factor), int(result_cropped.height * scaling_factor))
    result_cropped = result_cropped.resize(new_size, Image.ANTIALIAS) #todo: PIL.Image.Resampling.LANCZOS with ImageResampling or Image.LANCZOS if supported
    return result_cropped
