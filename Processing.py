#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 23:55:32 2018

@author: vickywinter
"""
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import sys

nltk.download('punkt')
nltk.download('stopwords')

def clean(all_data):
    all_data["tweet"]=all_data["tweet"].str.lower()
    #replace the hastag by its value
    all_data["tweet"]=all_data["tweet"].str.replace("#","")
    all_data["tweet"]=all_data["tweet"].str.replace("@user","USER_TAG")
    all_data["tweet"]=all_data["tweet"].str.replace("â¦"," Emo_neg ")
    all_data["tweet"]=all_data["tweet"].str.replace("â"," Emo_neg ")
    all_data["tweet"]=all_data["tweet"].str.replace("ð"," Emo_pos ")
    all_data["tweet"]=all_data["tweet"].str.replace("â","'")
    all_data["tweet"]=all_data["tweet"].str.replace("[ðâ]..."," Emo_pos ")

    all_data["tweet"]=all_data["tweet"].str.replace('(http|https|ftp)://[a-zA-Z0-9\\./]+','URL')
    all_data["tweet"]=all_data["tweet"].str.replace('?',' PUN_QUE ')
    all_data["tweet"]=all_data["tweet"].str.replace('!',' PUN_EXC ')
    removelist="'$%"
    all_data["tweet"]=all_data["tweet"].str.replace('[^\w\s'+removelist+']'," ")
    all_data["tweet"]=all_data["tweet"].str.replace('(.)\1{1,};','\1\1')
    
    
   
    
    all_data["tweet"]=all_data["tweet"].str.replace("  "," ")
    all_data['tweet']=all_data['tweet'].map(lambda x: x.strip())
    
    all_data['token']=all_data.apply(lambda row: row['tweet'].split(" "),axis=1)
    
    stopword_set = set(stopwords.words("english")) 
    negation_words=['never','no','nothing','nowhere','none', \
                'not','nor','havent',"haven't",'hasnt',"hasn't", \
                'hadnt',"hadn't",'cant',"can't",'cannot', "mightn't","mustn't","needn't","shan't", \
                'couldnt',"couldn't",'shouldnt',"shouldn't",'wasn', "wasn't",'weren',"weren't", \
                'wont','wouldnt',"wouldn't",'dont',"don't", \
                'doesnt',"doesn't",'didnt',"didn't",'isnt',"isn't", \
                'arent',"aren't",'aint']
    stopword_set=[x for x in stopword_set if x not in negation_words]
    all_data["token"]=all_data['token'].apply(lambda row: [x for x in row if x not in stopword_set])
    
    stemmer=nltk.stem.PorterStemmer()
    all_data["token"]=all_data['token'].apply(lambda row: [stemmer.stem(w) for w in row])
    all_data["token"]=all_data['token'].apply(lambda row: [x for x in row if len(x)>2])
    all_data["tweet"]=all_data['token'].apply(lambda x: ' '.join(x))
    all_data['sentiment']=all_data['label'].apply(lambda x: 'negative' if x==1 else 'positive')

    all_data=all_data[all_data['tweet']!=""]
    
    train_data=all_data[all_data['train/test']=='train']
    test_data=all_data[all_data['train/test']=='test']
    
    sys.stderr.write( '\ntraining data size  = '+str(len( train_data )) )
    sys.stderr.write( '\ntest data size  = '+str(len( test_data )) )
    return (all_data,train_data,test_data)



def data_fre(data,text,dia=False):
    res=pd.DataFrame(data['tweet'].str.split(expand=True).stack().value_counts())
    res.columns=['Count']
    res['Words']=res.index
    res['fre']=res['Count']/res['Count'].sum()
    if dia:
        plt.xticks(rotation='90')
        sns.barplot(x=res.head(30)['Words'],y=res.head(30)['fre'])
        plt.xlabel('Most frequency words been used in ' + text +' tweet')
        plt.ylabel('Frequency', multialignment='center')
    return res

def remove_netural_word(data,train):
    negative=data[data['label']==1]
    positive=train[train['label']==0]  

    all_data_fre=data_fre(data,'all')
    negative_fre=data_fre(negative,'negative')
    positive_fre=data_fre(positive,'positve')

    com=negative_fre.merge(positive_fre,how='outer',left_on=['Words'],right_on=['Words'])
    com['neg/pos']=com['fre_x']/com['fre_y']
    com=com.sort_values('neg/pos')
    com=com[com['neg/pos']>0]
    
    high_ratio_tail=com.tail(10)
    high_ratio_head=com.head(10)
    
    print( 'Words exist most in negative tweet   = '+high_ratio_tail['Words'] )
    print( 'Words exist most in positive tweet   = '+high_ratio_head['Words'])
    
    filtered = com[com['neg/pos'].apply(lambda x: x>0.995 and x<1.05)]
    remove=list(filtered['Words'])
    data['token']=data['token'].apply(lambda x: [i for i in x if i not in remove])
    data["tweet"]=data['token'].apply(lambda x: ' '.join(x))
    
    data=data[data['tweet']!=""]
    train_data=data[data['train/test']=='train']
    test_data=data[data['train/test']=='test']
    
    sys.stderr.write( '\ntraining data size  = '+str(len( train_data )) )
    sys.stderr.write( '\ntest data size  = '+str(len( test_data )) )
    return (data,train_data,test_data)