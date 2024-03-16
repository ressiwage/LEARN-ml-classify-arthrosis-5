import numpy as np
from PIL import Image
import debug, model

from matplotlib import pyplot as plt
import preprocess

def img_to_bytes(path):
    with open(path, "rb") as image:
        f = image.read()
        b = bytearray(f)
    return b

def predict(path, debug):
    result={}
    model_=model.model('65-relative-success-v1.weights.h5')
    preprocess_=preprocess.preprocess
    
    if debug:
        for ind, img in enumerate(debug.debug(model_, np.array(preprocess_(path)))):
            new_image = Image.fromarray(img)
            new_image.save(f'{path}_debug{ind}.png')
            result['debug_imgs'] = result.get('debug_imgs', []).append(f'{path}_debug{ind}.png')
    
    result['prediction']=model_.predict(np.array(preprocess_(path)))
    return result