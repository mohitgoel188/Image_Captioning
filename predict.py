from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras import Model
import numpy as np
import json
import heapq
from collections import deque
import sys
from keras.applications.inception_v3 import InceptionV3
from keras import Model
from keras.models import model_from_json
import cv2
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu,SmoothingFunction
from data_cleaning_images import preprocess

with open('Data/Processed/word2ind.json') as fh:
    word2ind = json.load(fh)
print('Loading Vocab...')
ind2word = list(word2ind.keys())
max_cap_length = 35
bleu_1 = (1,0,0,0) 
bleu_2 = (0,1,0,0) 
bleu_3 = (0,0,1,0) 
bleu_4 = (0,0,0,1) 
bleu = (bleu_1,bleu_2,bleu_3,bleu_4)
beam_width = 3
# print('Loading Inception V3...')
# iv3_model = InceptionV3(weights='imagenet')
# iv3_model = Model(iv3_model.input, iv3_model.layers[-2].output)

print('Loading Trained RNN Model...')
with open('Data/Saved Models/ImgCap_model40.json') as fh:
    model_json = json.load(fh)
model = model_from_json(model_json)
model.load_weights('Data/Saved Models/model_50.h5')

def greedySearch(encoded_img, reference=None, model=model):
    in_text = 'startseq'
    for i in range(max_cap_length):
        sequence = [word2ind[w] for w in in_text.split() if word2ind.get(w,None)]
        sequence = pad_sequences([sequence], maxlen=max_cap_length)
        yhat = model.predict([np.array(encoded_img),np.array(sequence)], verbose=0)
        yhat = np.argmax(yhat)
        word = ind2word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    predicted_seq = in_text.lstrip('startseq').rstrip('endseq')
    if reference:
        hypothesis = predicted_seq.split()
        bleu_score = sentence_bleu(list(map(lambda x: x.lstrip('startseq').rstrip('endseq').split(),reference)), hypothesis,smoothing_function=SmoothingFunction().method1)
        print(f'Bleu Score: {bleu_score}')
        for i in range(4):
            bleu_score = sentence_bleu(list(map(lambda x: x.lstrip('startseq').rstrip('endseq').split(),reference)), hypothesis,bleu[i],smoothing_function=SmoothingFunction().method1)
            print(f'Bleu Score (nGram-{i+1}): {bleu_score}')    
        
        
    return predicted_seq

def beamSearch(encoded_img, beam_width,reference=None, verbose=0, model=model):
    # in_text = 'startseq'
    beam = deque([(0,'startseq',1)])
    completed = []
    while len(beam)>0 :
        ini_prob,text,count = beam.popleft()
        if count<max_cap_length:
            seq = [word2ind[w] for w in text.split() if word2ind.get(w,None)]
            seq = pad_sequences([seq],maxlen = max_cap_length)
            yhat = model.predict([np.array(encoded_img),np.array(seq)],verbose = 0)
            yhat = list(yhat[0])
            yhat = [(np.log(prob),ind) for prob,ind in zip(yhat,range(len(yhat)))]
            for prob,ind in heapq.nlargest(3,yhat):
                predicted_word = ind2word[ind]
                if predicted_word == 'endseq' or count+1==max_cap_length:
                    final_prob = (ini_prob + prob)/count+1 
                    completed.append((final_prob,f'{text} {predicted_word}',count+1))
                    if verbose:
                        print(f'{final_prob} : {text} {predicted_word}')
                else:
                    beam.append((ini_prob + prob,f'{text} {predicted_word}',count+1))
            if len(beam)>beam_width**2:
                beam = deque(heapq.nlargest(3,beam))
    predicted_seq = max(completed)[1].lstrip('startseq').rstrip('endseq')
    if reference:
        hypothesis = predicted_seq.split()
        bleu_score = sentence_bleu(list(map(lambda x: x.lstrip('startseq').rstrip('endseq').split(),reference)), hypothesis,smoothing_function=SmoothingFunction().method1)
        print(f'Bleu Score: {bleu_score}')
        for i in range(4):
            bleu_score = sentence_bleu(list(map(lambda x: x.lstrip('startseq').rstrip('endseq').split(),reference)), hypothesis,bleu[i],smoothing_function=SmoothingFunction().method1)
            print(f'Bleu Score (nGram-{i+1}): {bleu_score}')    
    return predicted_seq
    
def main():
    basedir = 'Data/Images'

    if len(sys.argv)>1:
        verbose = int(sys.argv[1])
    else:
        verbose = 0

    print('Loading test set...')
    with open('Data/Processed/test.json') as fh:
        test_set = json.load(fh)

    hypothesises_greed=[]
    hypothesises_beam=[]
    references=[]
    i=0
    for image_filename,output in test_set.items():
        # print(output)
        references.append(list(map(lambda x: x.lstrip('startseq').rstrip('endseq').split(),output)))
        print(f'\nPredicting image {i+1}...')
        encoded_img = preprocess(f'{basedir}/{image_filename}').reshape((1,2048))
        
        print('Doing Greedy Search...')
        predicted_seq = greedySearch(encoded_img,output)
        hypothesises_greed.append(predicted_seq.split())
        print()
        print(f"\nGreedy: {predicted_seq}")

        print(f'\nDoing Beam Search with width={beam_width}...')    
        predicted_seq = beamSearch(encoded_img,beam_width,output,verbose)
        print(f"\nBeam: {predicted_seq}")
        hypothesises_beam.append(predicted_seq.split())

        if verbose:
            img = cv2.imread(f'{basedir}/{image_filename}')
            cv2.imshow(f'Image {i}',img)
            cv2.waitKey(10)
        i+=1
        # if i>2:
        #     break
    # for a,b in zip(hypothesises_greed,references):
    #     print(f'{a} -- {b}')
    # print('\n\n')
    # for a,b in zip(hypothesises_beam,references):
    #     print(f'{a} -- {b}')
    print(f'\n\nGreedy Corpus Score: {corpus_bleu(references,hypothesises_greed,smoothing_function=SmoothingFunction().method1)}')
    for i in range(4):
        print(f'Greedy Corpus Score (nGram-{i+1}): {corpus_bleu(references,hypothesises_greed,bleu[i],smoothing_function=SmoothingFunction().method1)}')
    
    print(f'\n\nBLEU Corpus Score: {corpus_bleu(references,hypothesises_beam,smoothing_function=SmoothingFunction().method1)}')
    for i in range(4):
        print(f'\nBLEU Corpus Score (nGram-{i+1}): {corpus_bleu(references,hypothesises_beam,bleu[i],smoothing_function=SmoothingFunction().method1)}')
if __name__ == "__main__":
    main()
    