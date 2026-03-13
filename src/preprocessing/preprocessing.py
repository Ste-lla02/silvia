from src.preprocessing.bands_split import red, blue, green, alpha
from src.preprocessing.alternative_ndvi import *

def plain(image, name):
    return image

splitting_broker = {
    'plain': plain,
    'red': red,
    'blue': blue,
    'green': green,
    'alpha': alpha,
    'ndvi_vari': vari_ndvi,
    'ndvi_tgi': tgi_ndvi,
    'ndvi_gli': gli_ndvi,
    'ndvi_ngrdi': ngrdi_ndvi,
    'ndvi_exg': exg_ndvi
}