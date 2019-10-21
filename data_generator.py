import json

import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def data_generator(img2cap, img2encd, wordtoix, max_cap_length, vocab_size, batchsize):
    x1, x2, y = list(), list(), list()
    n = 0
    # Infinite loop so that generator can continue yielding values after every epoch
    while True:
        for img_filename, cap_list in img2cap.items():
            n += 1
            # retrieve the image encoding 
            img_encoding = img2encd[img_filename]
            for cap in cap_list:
                # encode the sequence
                seq = [wordtoix[word] for word in cap.split(' ') if wordtoix.get(word, None)]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_cap_length)[0]
                    # encode output sequence
                    out_seq = to_categorical(out_seq, num_classes=vocab_size)
                    # store
                    x1.append(img_encoding)
                    x2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == batchsize:
                yield [[np.array(x1), np.array(x2)], np.array(y)]
                x1, x2, y = list(), list(), list()
                n = 0


if __name__ == "__main__":
    train_images = joblib.load('Data/Processed/encoded_trainImages.pkl')
    with open('Data/Processed/train.json') as fh:
        train_captions = json.load(fh)
    with open('Data/Processed/word2ind.json') as fh:
        word2ind = json.load(fh)
    gen = data_generator(train_captions, train_images, word2ind, 35, 1652, 2)
    print(next(gen))
