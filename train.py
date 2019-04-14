import joblib
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import json
import numpy as np
from keras import Input, layers, optimizers, Model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization 
from keras.optimizers import Adam, RMSprop
from keras.layers import add
import math
from data_generator import data_generator

max_cap_length = 35
vocab_size = 1652
embedding_dim = 200

# For running locally
train_images_path = 'Data/Processed/encoded_trainImages.pkl'
train_captions_path = 'Data/Processed/train.json'
test_images_path = 'Data/Processed/encoded_testImages.pkl'
test_captions_path = 'Data/Processed/test.json'
word2ind_path = 'Data/Processed/word2ind.json'
word_embedding_path = 'Data/Processed/word_embedding.pkl'
save_location = 'Data/Saved Models/'

# # For running in google colab
# from google.colab import drive
# drive.mount('/content/gdrive')

# train_images_path = 'gdrive/My Drive/Data/encoded_trainImages.pkl'
# train_captions_path = 'gdrive/My Drive/Data/train.json'
# test_images_path = 'gdrive/My Drive/Data/encoded_testImages.pkl'
# test_captions_path = 'gdrive/My Drive/Data/test.json'
# word2ind_path = 'gdrive/My Drive/Data/word2ind.json'
# word_embedding_path = 'gdrive/My Drive/Data/word_embedding.pkl'
# save_location = 'gdrive/My Drive/Saved Models/'

# Model Construction

# For Image input
inputs2 = Input(shape=(max_cap_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# For Captions
inputs2 = Input(shape=(max_cap_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Merging
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='elu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

embedding_matrix = joblib.load(word_embedding_path)
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss='categorical_crossentropy',optimizer='adam')

print(model.summary())

train_images=joblib.load(train_images_path)
with open(train_captions_path) as fh:
    train_captions = json.load(fh)
with open(word2ind_path) as fh:
    word2ind = json.load(fh)

epochs = 30
N_images_per_batch = 30
steps = math.ceil(len(train_captions)/N_images_per_batch)

generator = data_generator(train_captions, train_images, word2ind, max_cap_length, vocab_size, N_images_per_batch)
model.fit_generator(generator, epochs=epochs, steps_per_epoch=1, verbose=1, workers=-1)

model.optimizer.lr = 0.0001
N_images_per_batch = 300
steps = math.ceil(len(train_captions)/N_images_per_batch)
generator = data_generator(train_captions, train_images, word2ind, max_cap_length, vocab_size, N_images_per_batch)
model.fit_generator(generator, epochs=10, steps_per_epoch=steps, verbose=1, workers=-1)

with open(f'{save_location}ImgCap_model40.json','w') as fh:
    json.dump(model.to_json(),fh)
model.save_weights(f'{save_location}model_50.h5')
