from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import sys
import random
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from nlpaug.util import Action

def removeStopWords(text,sWords):
    return ' '.join([word for word in text.split() if word.lower() not in (sWords)])

def stem(text,stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def removeURL(text):
    urlPat=re.compile('https?://\S+|www\.\S+')
    text=urlPat.sub('',text)
    return text

def removeTags(text):
    tagsPat=re.compile('<.*?>')
    text=tagsPat.sub('',text)
    return text

def removeEmoji(text):
    emojiTag=re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF" 
    u"\U0001F680-\U0001F6FF" 
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
    text=emojiTag.sub('',text)
    return text

def removeBetweenBrackets(text):
    pat=re.compile('\[[^]]*\]')
    text=pat.sub('',text)
    return text

def encodeTarget(data,targetCols):
    df=data.copy()
    for i in targetCols:
        if df[i].dtypes=='object':
            encoder=LabelEncoder()
            df[i]=encoder.fit_transform(df[i])
    return df

def process(data):
    df=data.copy()
    stopWords=stopwords.words('english')
    snowBall=SnowballStemmer('english')
    for i in df.columns:
        if df[i].dtypes=='object':
            df[i]=df[i].apply(lambda x: removeTags(x))
            df[i]=df[i].apply(lambda x: removeURL(x))
            df[i]=df[i].apply(lambda x: removeEmoji(x))
            df[i]=df[i].apply(lambda x: removeBetweenBrackets(x))
            #df[i]=df[i].apply(lambda x: removeStopWords(x,stopWords))
            #df[i]=df[i].apply(lambda x: stem(x,snowBall))
    return df

def exportData(data,path):
    data.to_csv(path+'.csv',index=False)

def createAugmenter():
    aug=naf.Sometimes([
    naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased',action=Action.SUBSTITUTE),
    naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased',action=Action.INSERT)],
    aug_p=0.7)
    return aug
def main(dataPath,targets,export):
    df=pd.read_csv(dataPath)
    df=process(df)
    df=encodeTarget(df,targets)
    exportData(df,export)

def augmentText(text,labels,augmenter):
    augmentedText=[]
    augmentedSentiment=[]
    for i in range(len(text)):
        if random.random()<0.4:
            augmentedText.append(augmenter.augment(text[i]))
            augmentedSentiment.append(labels[i])
        else:
            continue
    return augmentedText,augmentedSentiment

def splitToSent(data):
    return sent_tokenize(data)     

if __name__=='__main__':
    args=sys.argv[1:]
    datsetPath=args[0]
    targetNames=args[1].split(',')
    exportPath=args[2]
    main(datsetPath,targetNames,exportPath)