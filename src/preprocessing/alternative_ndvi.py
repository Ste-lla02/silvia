import numpy as np
from PIL import Image
from src.preprocessing.bands_split import split_and_convert_all

epsilon = 1e-6

def vari_ndvi(image, name):
    VARI_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        VARI = (g - r) / (g + r - b + epsilon)
        VARI_normalized = (VARI - np.min(VARI)) / (np.max(VARI) - np.min(VARI))
        VARI_image = Image.fromarray((VARI_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"VARI Calculation Error with {name}: {e}")
    return VARI_image

def tgi_ndvi(image, name):
    TGI_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        TGI = -0.5 * (r - 0.39 * g - 0.61 * b)
        TGI_normalized = (TGI - np.min(TGI)) / (np.max(TGI) - np.min(TGI))
        TGI_image = Image.fromarray((TGI_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"TGI Calculation Error with {name}: {e}")
    return TGI_image

def gli_ndvi(image, name):
    GLI_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        numerator = 2 * g - r - b
        denominator = 2 * g + r + b + epsilon
        GLI = numerator / denominator
        GLI_normalized = (GLI - np.min(GLI)) / (np.max(GLI) - np.min(GLI))
        GLI_image = Image.fromarray((GLI_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"GLI Calculation Error with {name}: {e}")
    return GLI_image


def ngrdi_ndvi(image, name):
    NGRDI_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        numerator = g - r
        denominator = g + r + epsilon
        NGRDI = numerator / denominator
        NGRDI_normalized = (NGRDI - np.min(NGRDI)) / (np.max(NGRDI) - np.min(NGRDI))
        NGRDI_image = Image.fromarray((NGRDI_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"NGRDI Calculation Error with {name}: {e}")
    return NGRDI_image


def exg_ndvi(image, name):
    ExG_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        ExG = 2 * g - r - b
        ExG_normalized = (ExG - np.min(ExG)) / (np.max(ExG) - np.min(ExG))
        ExG_image = Image.fromarray((ExG_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"ExG Calculation Error with {name}: {e}")
    return ExG_image