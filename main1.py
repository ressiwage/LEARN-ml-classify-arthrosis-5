import numpy as np
import debug
import model
from matplotlib import pyplot as plt
import preprocess

model=model.model('65-relative-success-v1.weights.h5')

preprocess=preprocess.preprocess

img=np.array(preprocess(r'test\0\9003175R.png'))


for layer in model.layers:

    print(layer.name) 






# Display the final image
plt.imshow(debug.debug(model, np.array(preprocess(r'test\3\9011053L.png'))))
plt.show()

plt.imshow(debug.debug(model, np.array(preprocess(r'test\2\9008884L.png'))))
plt.show()

plt.imshow(debug.debug(model, np.array(preprocess(r'test\0\9003175R.png'))))
plt.show()

plt.imshow(debug.debug(model, np.array(preprocess(r'test\3\9092628R.png'))))
plt.show()

print(model.predict(np.array(
    preprocess(r'test\3\9011053L.png')+\
    preprocess(r'test\2\9008884L.png')+\
    preprocess(r'test\0\9003175R.png')+\
    preprocess(r'test\3\9092628R.png')+\
    preprocess(r'test\2\9059837L.png')
    )))
