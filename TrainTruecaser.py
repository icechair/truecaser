"""
This script trains the TrueCase System
"""
import nltk
import nltk.corpus
from nltk.corpus import brown
from nltk.corpus import reuters
import pickle
import string
import math
import nltk.data

from TrainFunctions import *
from EvaluateTruecaser import defaultTruecaserEvaluation


uniDist = nltk.FreqDist()
backwardBiDist = nltk.FreqDist() 
forwardBiDist = nltk.FreqDist() 
trigramDist = nltk.FreqDist() 
wordCasingLookup = {}


        
        
        
"""
There are three options to train the true caser:
1) Use the sentences in NLTK
2) Use the train.txt file. Each line must contain a single sentence. Use a large corpus, for example Wikipedia
3) Use Bigrams + Trigrams count from the website http://www.ngrams.info/download_coca.asp

The more training data, the better the results
"""
         
'''
# :: Option 1: Train it based on NLTK corpus ::
print("Update from NLTK Corpus")
NLTKCorpus = brown.sents()+reuters.sents()+nltk.corpus.semcor.sents()+nltk.corpus.conll2000.sents()+nltk.corpus.state_union.sents()
updateDistributionsFromSentences(NLTKCorpus, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
'''
# :: Option 2: Train it based the train.txt file ::
 #Uncomment, if you want to train from train.txt
print("Update from train.txt file")
sentences = []
for line in open('train.de.txt'):
    sentences.append(line.strip())
    
tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
updateDistributionsFromSentences(tokens, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
     
   
# :: Option 3: Train it based ngrams tables from http://www.ngrams.info/download_coca.asp ::    
""" #Uncomment, if you want to train from train.txt
print("Update Bigrams / Trigrams")
updateDistributionsFromNgrams('ngrams/w2.txt', 'ngrams/w3.txt', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
"""

with open('distributions.obj', 'wb') as f:
    pickle.dump(uniDist, f, protocol=2)
    pickle.dump(backwardBiDist, f, protocol=2)
    pickle.dump(forwardBiDist, f, protocol=2)
    pickle.dump(trigramDist, f, protocol=2)
    pickle.dump(wordCasingLookup, f, protocol=2)


        
# :: Correct sentences ::

defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
        
        



        
        