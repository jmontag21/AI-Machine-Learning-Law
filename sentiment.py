"""
Created on Thu Aug 22 14:49:41 2019

@author: MONTAGDEV
"""
import math

#sentiment analysis on sentences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyser = SentimentIntensityAnalyzer()
def sentiment_scores(sentence):
    snt = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(snt)))
    return snt
    

for i in range(len(sentences)):
    #print(print_sentiment_scores(sentences[i]['text'])['neu'])
    compound = sentiment_scores(sentences[i]['text'])['compound']
    compound = int(10*compound)
    #print(compound)
    tag = ""
    if compound >= 0:
        letter = chr(ord('A')+compound)
        #print(compound)
        tag = ("POS"+letter)
    else:
        letter = chr(ord('A')+ -1*compound)
        #print(compound)
        tag = ("NEG"+letter)
        
    sentences[i]['text'] = sentences[i]['text'] +  " " + tag
    print(sentences[i]['text'])
