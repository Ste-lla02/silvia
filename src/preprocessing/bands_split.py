import numpy as np

def split_and_convert_all(image):
    r, g, b, a = image.split()
    r = np.array(r, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    return r, g, b, a

def red(image, image_name):
    r, _, _, _ = image.split()
    return r

def green(image, image_name):
    _, g, _, _ = image.split()
    return g

def blue(image, image_name):
    _, _, b, _ = image.split()
    return b

def alpha(image, image_name):
    _, _, _, a = image.split()
    return a
