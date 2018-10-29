#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:19:13 2018

@author: vickywinter
"""
import nltk
import sys
negation_words=['never','no','nothing','nowhere','none', \
                'not','nor','havent',"haven't",'hasnt',"hasn't", \
                'hadnt',"hadn't",'cant',"can't",'cannot', "mightn't","mustn't","needn't","shan't", \
                'couldnt',"couldn't",'shouldnt',"shouldn't",'wasn', "wasn't",'weren',"weren't", \
                'wont','wouldnt',"wouldn't",'dont',"don't", \
                'doesnt',"doesn't",'didnt',"didn't",'isnt',"isn't", \
                'arent',"aren't",'aint']

def Get_words_features(words):
    bag={}
    words_uni=[ 'has(%s)'% ug for ug in words ]
    
    if word_num>1:
        words_bi=['has(%s)'% ' '.join(map(str,bg)) for bg in nltk.bigrams(words)]
    else:
        words_bi=[]
    
    if word_num>2:
        words_tri=['has(%s)'% ' '.join(map(str,bg)) for bg in nltk.trigrams(words)]
    else:
        words_tri=[]
        
    for f in words_uni+words_bi+words_tri:
        bag[f]=1
        #print(bag)
    return bag

 
def Get_Negation_features(words):
    neg=[w in negation_words for w in words]
    left=[0.0]*len(words)
    prev=0.0
    for i in range(0,len(words)):
        if neg[i]:
            prev=1.0
            left[i]=prev
            prev=max(0.0,prev-0.1)
    
    right=[0.0]*len(words)
    prev=0.0
    for i in reversed(range(0,len(words))):
        if (neg[i]):
            prev=0.0
            right[i]=prev
            prev=max(0.0,prev-0.1)
    return dict( zip(['neg_l('+w+')' for w in  words] + ['neg_r('+w+')' for w \
           in  words],left + right ) )
        

    
def extract_features(words):
    features={}
    
    word_fes=Get_words_features(words)
    features.update(word_fes)  
    
    if neg:
        neg_fes=Get_Negation_features(words)
        features.update(neg_fes)
    #sys.stderr.write( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets' )
    return features
    
    
def get_train_test(train_tweet,test_tweet,word_num,neg):
    unigrams_fd = nltk.FreqDist()
    n_grams_fd=nltk.FreqDist()
    for( token, sentiment ) in train_tweet:
        words_uni = token 
        unigrams_fd.update(token)
        
        if word_num==2:
            words_bi=[' '.join(map(str,bg)) for bg in nltk.bigrams(token)]
            n_grams_fd.update(words_bi)
            
        if word_num==3:
            words_tri=[' '.join(map(str,bg)) for bg in nltk.trigrams(token)]
            n_grams_fd.update(words_tri)
    sys.stderr.write( '\nlen( unigrams ) = '+str(len( unigrams_fd.keys() )) )
    all_words=unigrams_fd.keys()
 
    if word_num>1:
        sys.stderr.write( '\nlen( n_grams ) = '+str(len( n_grams_fd )) )
        all_words = [ k for (k,v) in n_grams_fd.items() if v>1]
        sys.stderr.write( '\nlen( ngrams_sorted ) = '+str(len( all_words )) )
    #return all_words

    #extract_features(train_list)
    v_train = nltk.classify.apply_features(extract_features,train_tweet)
    v_test = nltk.classify.apply_features(extract_features,test_tweet)
    return(v_train,v_test)
    
word_num=1
neg=False
#a=pd.DataFrame(v_train)

