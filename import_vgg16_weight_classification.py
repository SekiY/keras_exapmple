from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys

if len(sys.argv) != 2:
    print("usage: python test_vgg16.py [image file]")
    sys.exit(1)

filename = sys.argv[1]

model = VGG16(weights='imagenet')

img = image.load_img(filename, target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

preds = model.predict(preprocess_input(x))
results = decode_predictions(preds, top=5)[0]
for result in results:
    print(result)
