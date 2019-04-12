import cv2
from keras.applications.inception_v3 import preprocess_input
import numpy as np

def preprocess(image_path, model):
    """ Convert all the images to size 299x299 as expected by the inception v3 model
        and returns the predicted feature vector of dim : (2048,1)

        Note:   model parameter expects model to be :-
                    from keras.applications.inception_v3 import InceptionV3
                    from keras import Model
                    model=InceptionV3(weights='imagenet')
                    model = Model(model.input, model.layers[-2].output)
    """            
    img = cv2.imread(image_path)
    img = cv2.resize(img,(299,299),interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(img, axis=0)
    feature_vector = model.predict(preprocess_input(img))
    return np.reshape(feature_vector, (2048,))
