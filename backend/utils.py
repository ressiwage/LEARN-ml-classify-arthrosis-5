import numpy as np
from PIL import Image
import debug, model
import os, errno

from matplotlib import pyplot as plt
import preprocess

import os, errno

def silentremove(filename):
    return
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def predict(path, dbg, features, brightness):
    result={}
    model_=model.model('65-relative-success-v1.weights.h5')
    preprocess_=preprocess.preprocess
    if dbg:
        for ind, img in enumerate(
                debug.debug(model_, np.array(preprocess_(path)), dbg=features, brightness=brightness)
            ):
            new_image = Image.fromarray(img)
            new_image.save(f'static/output/{path.split(".")[-2]}_debug{ind}.png')
            result['debug_imgs'] = result.get('debug_imgs', []) + [f'{path.split(".")[-2]}_debug{ind}.png']
    
    result['prediction']=model_.predict(np.array(preprocess_(path)))
    return result