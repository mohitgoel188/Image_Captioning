import string
from collections import Counter,defaultdict
import json
import numpy as np
import joblib
from os import makedirs

#Global Var
word_count_threshold=10

with open('Data/Flickr_TextData/Flickr8k.token.txt') as fh:
    imgCap={}
    table=str.maketrans('','',string.punctuation)
    max_cap_len=None
    word_count=defaultdict(int)
    #     c=0
    for line in fh:
        # Extracting caption
        imgFile,caption=line.split('\t')
        # Basic cleaning: lowercasing,removing punctuations and making list of words
        caption=caption.strip().lower().translate(table).split()
        caption=[word for word in caption if len(word)>1]
        # Removing alpha-numeric char
        caption=[word for word in caption if word.isalpha()]
        # Making vocabulary
        for word in caption:
            word_count[word]    #word_count.get(word,0)+1
        # Finding the maximum caption length
        if not max_cap_len or max_cap_len<len(caption) : 
            max_cap_len=len(caption)
        # Making a string from list of words
        caption=' '.join(caption)
        
        # Extracting image file name
        imgFile=imgFile.split('#')[0]
        if not imgCap.get(imgFile,None):
            imgCap[imgFile]=[]
        imgCap[imgFile].append(f"startseq {caption} endseq")
#         c+=1
#         if c>6:
#             break

# Due to 'startseq' and 'endseq'      
max_cap_len+=2 

# Train Test CV
with open('Data/Flickr_TextData/Flickr_8k.trainImages.txt') as fh:
    train={}
    for line in fh:
        train[line.strip()]=imgCap[line.strip()]

with open('Data/Flickr_TextData/Flickr_8k.testImages.txt') as fh:
    test={}
    for line in fh:
        test[line.strip()]=imgCap[line.strip()]

with open('Data/Flickr_TextData/Flickr_8k.devImages.txt') as fh:
    cross_validation={}
    for line in fh:
        cross_validation[line.strip()]=imgCap[line.strip()]

makedirs('Data/Processed',exist_ok=True)
with open('Data/Processed/corpus.json','w') as fh:
    json.dump(imgCap,fh)
with open('Data/Processed/train.json','w') as fh:
    json.dump(train,fh)
with open('Data/Processed/test.json','w') as fh:
    json.dump(test,fh)
with open('Data/Processed/cross_validation.json','w') as fh:
    json.dump(cross_validation,fh)

# Reducing vocab
for key,val in train.items():
    for cap in val:
        for word in cap.split():
            word_count[word]+=1
del_key=[]
# word_count_threshold=10
for key,val in word_count.items():
    if val<word_count_threshold:
        del_key.append(key)
for key in del_key:
    word_count.pop(key)
vocab=[0,*word_count.keys()]

# Word2Ind and Ind2Word
word2ind={}
ind2word=vocab
for ind,word in enumerate(vocab):
    word2ind[word]=ind
with open('Data/Processed/word2ind.json','w') as fh:
    json.dump(word2ind,fh)

# Word Embedding - GLoVe
embeddings_index = {}
with open('glove.6B.200d.txt', encoding="utf-8") as fh:
    for line in fh:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

# Get 200-dim dense vector for each word in vocabulary
with open('Data/Processed/word2ind.json') as fh:
    word2ind = json.load(fh)
embedding_matrix = np.zeros((len(word2ind),200))
for word, i in word2ind.items():
    embedding_vector = embeddings_index.get(word)
    # Words not found in the embedding_index will be all zeros
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Storing the embedding
joblib.dump(embedding_matrix,'Data/Processed/word_embedding.pkl')