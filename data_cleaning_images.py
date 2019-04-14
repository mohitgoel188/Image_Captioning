import cv2
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import joblib
import numpy as np
from time import process_time

print('Loading Inception V3...')
model=InceptionV3(weights='imagenet')
model = Model(model.input, model.layers[-2].output)

def preprocess(image_path, model=model):
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

if __name__ == "__main__":    
    dirpath = 'Data/Flickr_Images/'
    for set_type in ['test','dev','train']:
        with open(f'Data/Flickr_TextData/Flickr_8k.{set_type}Images.txt') as fh:
            encoded={}
            t1=process_time()
            for img_filename in fh:
                img_filename = img_filename.strip()
                feature_vector = preprocess(f"{dirpath}{img_filename}",model)
                encoded[img_filename]=feature_vector
            print(f'Time taken in encoding {set_type} images: {process_time()-t1}')
            joblib.dump(encoded,f'Data/Processed/encoded_{set_type}Images.pkl')
        