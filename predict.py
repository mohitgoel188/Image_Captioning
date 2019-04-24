""" This module predicts the caption of the image as per arguments.
    
    usage: predict.py [-h] [--image IMAGE] [--show_image SHOW_IMAGE]
                  [--verbose VERBOSE] [--cam CAM] [--eval EVAL]
                  [--desc_json DESC_JSON]

    optional arguments:
    -h, --help                                  show this help message and exit
    --image IMAGE, -i IMAGE                     folder location of images
    --show_image SHOW_IMAGE, -s SHOW_IMAGE      show image with caption too
    --verbose VERBOSE, -v VERBOSE               for detailed output of beam search
    --cam CAM, -c CAM                           for capturing image from web cam
    --eval EVAL, -e EVAL                        for evaluation and finding bleu score
    --desc_json DESC_JSON, -d DESC_JSON         json filepath mapping image to captions if eval==1

"""
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
# import matplotlib.pyplot as plt
import argparse
from glob2 import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', help="folder location of images", type= str, default= 'Data/Flickr_Images')
parser.add_argument('--show_image', '-s', help="show image with caption too", type= int, default= 1)
parser.add_argument('--verbose', '-v', help="for detailed output of beam search", type= int, default= 0)
parser.add_argument('--cam', '-c', help="for capturing image from web cam", type= int, default= 0)
parser.add_argument('--eval', '-e', help="for evaluation and finding bleu score", type= int, default= 0)
parser.add_argument('--desc_json', '-d', help="json filepath mapping image to captions if evalution==1", type= str, default= 'Data/Processed/test.json')
args = parser.parse_args()

print(parser.format_help())  
print(f"Using Arguments: \n Image Dir: {args.image}")
print(f'Showing Image: {args.show_image}')
print(f"Verbose: {args.verbose}")
print(f'Using web cam: {args.cam}')
if args.eval:
    print(f'Finding bleu score using: {args.desc_json}')

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
with open('Data/Saved Models/ImgCap_model_40.json') as fh:
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
                    if args.verbose:
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
    if args.eval:
        try:
            basedir = args.image

            print('Loading test set...')
            with open(args.desc_json) as fh:
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
                greedy_predicted_seq = greedySearch(encoded_img,output)
                hypothesises_greed.append(greedy_predicted_seq.split())
                print()
                print(f"\nGreedy: {greedy_predicted_seq}")

                print(f'\nDoing Beam Search with width={beam_width}...')    
                beam_predicted_seq = beamSearch(encoded_img,beam_width,output,args.verbose)
                print(f"\nBeam: {beam_predicted_seq}")
                hypothesises_beam.append(beam_predicted_seq.split())

                print('\nActual :')
                for cap in references[-1]:
                    print(' '.join(cap))

                if args.show_image:
                    img = cv2.imread(f'{basedir}/{image_filename}')
                    cv2.imshow(f'Image {i}',img)
                    key = cv2.waitKey(3000)
                    if key == 27:   #if ESC is pressed, exit loop
                        cv2.destroyAllWindows()
                    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # plt.show()
                i+=1
                # if i>5:
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
                print(f'BLEU Corpus Score (nGram-{i+1}): {corpus_bleu(references,hypothesises_beam,bleu[i],smoothing_function=SmoothingFunction().method1)}')
        except KeyboardInterrupt:
            print(f'\n\nGreedy Corpus Score: {corpus_bleu(references,hypothesises_greed,smoothing_function=SmoothingFunction().method1)}')
            for i in range(4):
                print(f'Greedy Corpus Score (nGram-{i+1}): {corpus_bleu(references,hypothesises_greed,bleu[i],smoothing_function=SmoothingFunction().method1)}')
            
            print(f'\n\nBLEU Corpus Score: {corpus_bleu(references,hypothesises_beam,smoothing_function=SmoothingFunction().method1)}')
            for i in range(4):
                print(f'BLEU Corpus Score (nGram-{i+1}): {corpus_bleu(references,hypothesises_beam,bleu[i],smoothing_function=SmoothingFunction().method1)}')        
            exit()
    elif args.cam:
        cam = cv2.VideoCapture(0)
        # cv2.namedWindow("Press space to take snapshot")
        while True:
            ret, frame = cam.read()
            cv2.imshow("Press space to take snapshot", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            if k%256 == 32: # Take snapshot if SPACE pressed
                break
        cam.release()

        cv2.imshow('Snapshot',frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        try:
            found = glob('\Data\Captured\*.jpg')[-1].split('\\')[-1].split('.')[0]
            id = int(found) + 1
        except:
            id = 1        
        if not os.path.exists(f'{os.getcwd()}/Data/Snapshots'):
            os.mkdir(f'{os.getcwd()}/Data/Snapshots')
        cv2.imwrite(f'{os.getcwd()}/Data/Snapshots/Shot_{id}.jpg',frame)
        print(f'\nPredicting image...')
        encoded_img = preprocess(img=frame).reshape((1,2048))
        
        print('Doing Greedy Search...')
        greedy_predicted_seq = greedySearch(encoded_img)
        print()
        print(f"\nGreedy: {greedy_predicted_seq}")

        print(f'\nDoing Beam Search with width={beam_width}...')    
        beam_predicted_seq = beamSearch(encoded_img,beam_width,verbose=args.verbose)
        print(f"\nBeam: {beam_predicted_seq}")

        if args.show_image:
            cv2.imshow(f'Image',frame)
            key = cv2.waitKey(3000)
            if key == 27:   #if ESC is pressed, exit loop
                cv2.destroyAllWindows()
    else:
        image_paths = glob(f'{args.image}/*.jpg')
        i=0
        for image_path in image_paths:

            print(f'\nPredicting image {i+1}...')
            encoded_img = preprocess(image_path).reshape((1,2048))
            
            print('Doing Greedy Search...')
            greedy_predicted_seq = greedySearch(encoded_img)
            print()
            print(f"\nGreedy: {greedy_predicted_seq}")

            print(f'\nDoing Beam Search with width={beam_width}...')    
            beam_predicted_seq = beamSearch(encoded_img,beam_width,verbose=args.verbose)
            print(f"\nBeam: {beam_predicted_seq}")

            if args.show_image:
                img = cv2.imread(image_path)
                cv2.imshow(f'Image {i}',img)
                key = cv2.waitKey(3000)
                if key == 27:   #if ESC is pressed, exit loop
                    cv2.destroyAllWindows()
                # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.show()
            i+=1

if __name__ == "__main__":
    main()
    