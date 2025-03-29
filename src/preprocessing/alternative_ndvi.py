import numpy as np
from PIL import Image
from src.preprocessing.bands_split import split_and_convert_all

#todo: clean the comments
def vari_ndvi(image, name):
    VARI_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del VARI
        # Evitiamo divisioni per zero aggiungendo un valore epsilon molto piccolo
        epsilon = 1e-6 #todo: estraiamolo come parametro di configurazione
        VARI = (g - r) / (g + r - b + epsilon)
        # normalization for a better visualisation
        VARI_normalized = (VARI - np.min(VARI)) / (np.max(VARI) - np.min(VARI))
        # conversion
        VARI_image = Image.fromarray((VARI_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"VARI Calculation Error with {name}: {e}")
    return VARI_image

def tgi_ndvi(image, name):
    TGI_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        # TGI index computing
        TGI = -0.5 * (r - 0.39 * g - 0.61 * b)
        # Normalizza i valori del TGI per una migliore visualizzazione (opzionale)
        TGI_normalized = (TGI - np.min(TGI)) / (np.max(TGI) - np.min(TGI))
        # Converti in immagine e salva
        TGI_image = Image.fromarray((TGI_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"TGI Calculation Error with {name}: {e}")
    return TGI_image

"""#GLI function"""
def gli_ndvi(image, name):
    GLI_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del GLI
        numerator = 2 * g - r - b
        denominator = 2 * g + r + b + 1e-6  # Aggiungiamo un piccolo epsilon per evitare divisioni per zero
        GLI = numerator / denominator
        # Normalizza i valori del GLI per la visualizzazione (opzionale)
        GLI_normalized = (GLI - np.min(GLI)) / (np.max(GLI) - np.min(GLI))
        # Converti in immagine e salva
        GLI_image = Image.fromarray((GLI_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"GLI Calculation Error with {name}: {e}")
    return GLI_image


"""#NGRDI function"""
def ngrdi_ndvi(image, name):
    NGRDI_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del NGRDI
        numerator = g - r
        denominator = g + r + 1e-6  # Aggiungiamo un piccolo epsilon per evitare divisioni per zero
        NGRDI = numerator / denominator
        # Normalizza i valori del NGRDI per la visualizzazione (opzionale)
        NGRDI_normalized = (NGRDI - np.min(NGRDI)) / (np.max(NGRDI) - np.min(NGRDI))
        # Converti in immagine e salva
        NGRDI_image = Image.fromarray((NGRDI_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"NGRDI Calculation Error with {name}: {e}")
    return NGRDI_image


"""#ExG function"""
def exg_ndvi(image, name):
    ExG_image = None
    r, g, b, a = split_and_convert_all(image)
    try:
        # Calcolo del ExG
        ExG = 2 * g - r - b
        # Normalizza i valori del ExG per la visualizzazione (opzionale)
        ExG_normalized = (ExG - np.min(ExG)) / (np.max(ExG) - np.min(ExG))
        # Converti in immagine e salva
        ExG_image = Image.fromarray((ExG_normalized * 255).astype(np.uint8))
    except Exception as e:
        print(f"ExG Calculation Error with {name}: {e}")
    return ExG_image