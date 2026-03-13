from PIL import Image, ImageDraw
from src.utils.configuration import Configuration
from src.utils.utils import adjust_coordinate_rectangle


def crop_image_with_polygon(image, configuration):
    points = configuration.get('areaofinterest_image')
    scaling_factor = configuration.get("image_scaling")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255)

    result = Image.new("RGBA", image.size)
    result.paste(image, mask=mask)

    x_min, x_max, y_min, y_max = adjust_coordinate_rectangle(points)
    result_cropped = result.crop((x_min, y_min, x_max, y_max))

    new_size = (int(result_cropped.width * scaling_factor), int(result_cropped.height * scaling_factor))
    result_cropped = result_cropped.resize(new_size, Image.LANCZOS)
    return result_cropped
