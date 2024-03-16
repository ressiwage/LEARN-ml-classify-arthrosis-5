from keras.models import Model
from PIL import Image
import cv2, random
import numpy as np

def debug(model, img, dbg=False, brightness=1):
    new_model=Model(inputs=model.inputs, outputs=model.layers[8].output)

    feature_maps = new_model.predict(img)
    result=[]

    square = 8
    ix = 1
    images = []
    for _ in range(4):
        for _ in range(8):
            images.append(feature_maps[0, :, :, ix-1])        
            ix += 1

    # Assuming 'images' is a list of grayscale images
    colored_images = []
    for img_ in images:
        # Create a random color filter
        filter = np.zeros((37, 37, 3), dtype=np.float32)
        filter[:, :, 0] = 1  # Red channel
        filter[:, :, 1] = 0.1  # Green channel
        filter[:, :, 2] = 1  # Blue channel

        # Apply the color filter to the image
        colored_img = cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR) * filter

        # Append the colored image to the list
        colored_images.append(colored_img)

    # Resize images to 224x224
    resized_images = [cv2.resize(img_, (224, 224), interpolation=cv2.INTER_LINEAR) for img_ in colored_images]
    
    # Load the background image and resize it to 224x224
    background = cv2.resize(img[0]*255, (224, 224), interpolation=cv2.INTER_LINEAR)
    if dbg:
        result.extend([(cv2.addWeighted(background.astype(float), 0.5, i.astype(float)*16*brightness, 0.5, 0)).astype('uint8') for i in resized_images])

    # Ensure the background is in the correct color space
    if len(background.shape) == 2:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    resized_images = [sum(resized_images)]

    # Add each image to the background with opacity 0.1
    for img in resized_images:
        # We need to ensure that the images are in the correct format for this operation
        img = img.astype(float)
        background = background.astype(float)
        
        # Add the image to the background with opacity 0.1
        
        added_image = cv2.addWeighted(background, 0.5, cv2.multiply(background, img)/(16/brightness), 0.5, 0)

        # Convert the result back to 8-bit format
        background = np.clip(added_image, 0, 255).astype('uint8')
    result = [background] + result
    return result