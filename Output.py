#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 23:56:08 2018

@author: vickywinter
"""

#loading daa

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import decimal
import numpy as np
from nltk.classify import NaiveBayesClassifier
from sklearn.cross_validation import train_test_split
import sys
from sklearn.svm import LinearSVC
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix,accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
sys.path.append('/Users/vickywinter/Documents/NYC/Twitter/Script') 

import Processing as pr
import train_test_data as tt
import cnn as cn

test=pd.read_csv('/Users/vickywinter/Documents/NYC/Twitter/Data/test.csv',encoding='utf-8')
train=pd.read_csv('/Users/vickywinter/Documents/NYC/Twitter/Data/train.csv',encoding='utf-8')
test['train/test']='test'
train['train/test']='train'
all_data=pd.concat([train,test])
#all_data_1=all_data

def tweet_sentiment(all_data,test,train):
    all_data,train_data,test_data=pr.clean(all_data)
    all_data,train_data,test_data=pr.remove_netural_word(all_data,train_data)


    negative=all_data[all_data['label']==1]
    positive=train_data[train_data['label']==0]  

    neg=pr.data_fre(negative,'negative',True)
    pos=pr.data_fre(positive,'positive',True)


    train_tweet=[]
    for index, row in train_data.iterrows():
        train_tweet.append((row['token'],row['sentiment']))
        test_tweet=[]
        for index, row in test_data.iterrows():
            test_tweet.append((row['token'],False))

    v_train,v_test=tt.get_train_test(train_tweet,test_tweet,word_num,neg)
    v_test=[item[0] for item in v_test]
    classifier_NB = nltk.classify.NaiveBayesClassifier.train(v_train)
    classifier_NB.show_most_informative_features(20)
    nltk.classify.accuracy(classifier_NB, v_train)
    guess_NB = [classifier_NB.classify(x) for x in v_test]
    guess_NB=pd.DataFrame(guess_NB)
    guess_NB['NB_prediction']=guess_NB.apply(lambda x :  1 if x[0]=='negative' else 0, axis=1)
    guess_NB=guess_NB.drop(columns=0)

    v_train_1,v_test_1=tt.get_train_test(train_tweet,train_tweet,word_num,neg)
    v_test_1=[item[0] for item in v_test_1]
    guess_NB_train = [classifier_NB.classify(x) for x in v_test_1]
    guess_NB_train=pd.DataFrame(guess_NB_train)
    guess_NB_train['NB_prediction']=guess_NB_train.apply(lambda x :  1 if x[0]=='negative' else 0, axis=1)
    guess_NB_train=guess_NB_train.drop(columns=0)

    classifier_svm = nltk.classify.SklearnClassifier(LinearSVC())
    classifier_svm.train(v_train)
    nltk.classify.accuracy(classifier_svm, v_train)
    guess_SVM = [classifier_svm.classify(x) for x in v_test]
    guess_SVM=pd.DataFrame(guess_SVM)
    guess_SVM['SVM_prediction']=guess_SVM.apply(lambda x :  1 if x[0]=='negative' else 0, axis=1)
    guess_SVM=guess_SVM.drop(columns=0)

    v_train_1,v_test_1=tt.get_train_test(train_tweet,train_tweet,word_num,neg)
    v_test_1=[item[0] for item in v_test_1]
    guess_SVM_train = [classifier_svm.classify(x) for x in v_test_1]
    guess_SVM_train=pd.DataFrame(guess_SVM_train)
    guess_SVM_train['SVM_prediction']=guess_SVM_train.apply(lambda x :  1 if x[0]=='negative' else 0, axis=1)
    guess_SVM_train=guess_SVM_train.drop(columns=0)

    guess_cnn,guess_cnn_train=cn.cnn(train_data,test_data)

    guess_cnn_train=guess_cnn_train.drop(columns=[0,1])
    train_data=train_data.reset_index()
    train_result=pd.concat([train_data,guess_NB_train,guess_SVM_train, guess_cnn_train],axis=1)
    train_result=train_result.drop(columns=['train/test','sentiment'])
    train_result['predict_result']=train_result.apply(lambda x: 1 if x['NB_prediction']+x['SVM_prediction']+x['CNN_prediction']>1 else 0, axis=1)
    confusion_matrix(train_result['label'], train_result['predict_result'])
    accuracy=accuracy_score(train_result['label'], train_result['predict_result'])
    print('Training data accuracy %.5f ' % accuracy)

    guess_cnn=guess_cnn.drop(columns=[0,1])
    test_data=test_data.reset_index()
    test_result=pd.concat([test_data,guess_NB,guess_SVM, guess_cnn], axis=1)
    test_result=test_result.drop(columns=['label','train/test','sentiment'])
    test_result['predict_result']=test_result.apply(lambda x: 1 if x['NB_prediction']+x['SVM_prediction']+x['CNN_prediction']>1 else 0, axis=1)

    result = pd.merge(test, test_result, how='left', on=['id'])
    result=result.drop(columns=['train/test','index','date_y','NB_prediction','SVM_prediction','CNN_prediction'])
    result.columns=['id','tweet','date','tweet_sim','token','prediction']
    count=pd.value_counts(result['prediction'].values, sort=False)
    print('Size of predicted positive tweet %d ' % count[0])
    print('Size of predicted negative tweet %d ' % count[1])
    
    return result

result=tweet_sentiment(all_data,test,train)
writer = pd.ExcelWriter('/Users/vickywinter/Documents/NYC/Twitter/Data/output.xlsx')
result.to_excel(writer,'Sheet1')
writer.save()





'''
guess_cnn_train=guess_cnn_train.drop(columns=[0,1])
train_data=train_data.reset_index()
train_result=pd.concat([train_data,guess_NB_train,guess_SVM_train, guess_cnn_train],axis=1)
train_result=train_result.drop(columns=['train/test','sentiment'])
train_result['predict_result']=train_result.apply(lambda x: 1 if x['NB_prediction']+x['SVM_prediction']+x['CNN_prediction']>1 else 0, axis=1)
confusion_matrix(train_result['label'], train_result['predict_result'])
accuracy_score(train_result['label'], train_result['predict_result'])

train_result['predict_result']=train_result.apply(lambda x: 1 if x['SVM_prediction']+x['CNN_prediction']>1 else 0, axis=1)
confusion_matrix(train_result['label'], train_result['predict_result'])
accuracy_score(train_result['label'], train_result['predict_result'])








result={'Naive Bayes':(w1,w4,w2,w3) , 'SVM':(s1,s4,s2,s3)}
result=pd.DataFrame(result)
result.rename(index={0:'Single_Word',1:'Single_Word+Non',2:'Two_Words',3:'Three_Words'}, inplace=True)
result.plot(kind='bar')
plt.xticks(rotation='0')
plt.show()

y_testtt
y_test_cnn=y_testtt.drop(columns=[0,1])
test_result=pd.concat([test_data,guess_NB,guess_SVM, y_testtt], axis=1)

result = pd.merge(test, test_result, how='left', on=['id'])

writer = pd.ExcelWriter('/Users/vickywinter/Documents/NYC/Twitter/Data/output.xlsx')
result.to_excel(writer,'Sheet1')
writer.save()


writer = pd.ExcelWriter('/Users/vickywinter/Documents/NYC/Twitter/Data/output_token.xlsx')
test_data.to_excel(writer,'Sheet1')
writer.save()
