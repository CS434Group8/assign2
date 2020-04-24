import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
import copy
import time
import math 
import matplotlib.pyplot as plt 
import multiprocessing as mp


# Importing the dataset
imdb_data = pd.read_csv('IMDB.csv', delimiter=',')

whole_x_arr=imdb_data['review']
whole_y_arr=pd.read_csv('IMDB_labels.csv',delimiter=',')['sentiment']


train_x=whole_x_arr[:30000].values.tolist()
train_y=whole_y_arr[:30000].values.tolist()

valid_x=whole_x_arr[30000:40000].values.tolist()
valid_y=whole_y_arr[30000:].values.tolist()


test_x=whole_x_arr[40000:].values.tolist()


def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(r'\d+', "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

def getPy(Xarr,Yarr):
    positive_arr=[]
    negative_arr=[]
    num_positive_y=0
    index=0
    for y in Yarr:
        if(y=='positive'):
            num_positive_y+=1
            positive_arr.append(Xarr[index])
        else:
            negative_arr.append(Xarr[index])
        index+=1
    p1=round(num_positive_y/len(Yarr),3)
    p0=1-p1
    
    return p1,p0,positive_arr,negative_arr


def countWords(Xarr,alpha,vocabulary,defaultWordDict,vectorizer):
    vectorizer.fit(Xarr)
    words = {k:v for k, v in vectorizer.vocabulary_.items()}
    word_dict=copy.deepcopy(defaultWordDict)
    totalwords=0
    # print(len(vocabulary),len(word_dict))
    for key in word_dict:
        if(key in words):
            totalwords+=words[key]
            word_dict[key]=words[key]
    for word in word_dict:
        val=word_dict[word]+alpha
        word_dict[word]=val/(totalwords+len(vocabulary)*alpha)
    return word_dict

def predictY(Xarr,p1,p0,probablity_positive_dict,probablity_negative_dict,vocabulary,vectorizer):
    words=vectorizer.fit_transform(Xarr)
    faetures=vectorizer.get_feature_names()
    arr=words.toarray()
    result=[]
    
    count=0
    for row in arr:
        index=0
        pos_val=p1
        neg_val=p0
        for val in row:
            if val!=0 and faetures[index] in vocabulary:
                pos_val+=val*math.log(probablity_positive_dict[faetures[index]])
                neg_val+=val*math.log(probablity_negative_dict[faetures[index]])
            index+=1
        if(pos_val>neg_val):
            result.append(1)
        else:
            result.append(0)
        count+=1
    return result

def calAccuracy(predict_y,y):
    right=0
    for i in range(0,len(predict_y)):
        if(predict_y[i]==1 and y[i]=='positive'):
            right+=1
        if(predict_y[i]==0 and y[i]=='negative'):
            right+=1
    return (right/len(predict_y))


def tuneMaxFeatures(max_features_test):
    print('max_features_test: '+str(max_features_test))
    newvectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=max_features_test
    )
    # fit the vectorizer on the text
    words=newvectorizer.fit(imdb_data['review'])
    # get the vocabulary
    inv_vocab = {v: k for k, v in newvectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    train_p1,train_p0,train_positive_Xarr,train_negative_Xarr=getPy(train_x,train_y)
    defaultWordDict={key: 0 for key in vocabulary}
    
    train_positive_word_probablity=countWords(train_positive_Xarr,0.2,vocabulary,defaultWordDict,newvectorizer)
    train_negative_word_probablity=countWords(train_negative_Xarr,0.2,vocabulary,defaultWordDict,newvectorizer)
    valid_predict_Y=predictY(valid_x,train_p1,train_p0,train_positive_word_probablity,train_negative_word_probablity,vocabulary,newvectorizer)
    valid_accuracy=calAccuracy(valid_predict_Y,valid_y)
    return valid_accuracy

def tuneAplpha(alpha):
    global train_positive_Xarr,train_negative_Xarr,vocabulary,defaultWordDict,vectorizer
    print('alpha: '+str(alpha))
    train_positive_word_probablity=countWords(train_positive_Xarr,alpha,vocabulary,defaultWordDict,vectorizer)
    train_negative_word_probablity=countWords(train_negative_Xarr,alpha,vocabulary,defaultWordDict,vectorizer)

    valid_predict_Y=predictY(valid_x,train_p1,train_p0,train_positive_word_probablity,train_negative_word_probablity,vocabulary,vectorizer)
    valid_accuracy=calAccuracy(valid_predict_Y,valid_y)
    
    return valid_accuracy

def tuneBoth(max_features_test,alpha):
    # print('max_features_test: '+str(max_features_test)+' alpha: '+str(alpha))
    newvectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=max_features_test,
    )
    # fit the vectorizer on the text
    words=newvectorizer.fit(imdb_data['review'])
    # get the vocabulary
    inv_vocab = {v: k for k, v in newvectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    train_p1,train_p0,train_positive_Xarr,train_negative_Xarr=getPy(train_x,train_y)
    defaultWordDict={key: 0 for key in vocabulary}
    
    train_positive_word_probablity=countWords(train_positive_Xarr,alpha,vocabulary,defaultWordDict,newvectorizer)
    train_negative_word_probablity=countWords(train_negative_Xarr,alpha,vocabulary,defaultWordDict,newvectorizer)
    valid_predict_Y=predictY(valid_x,train_p1,train_p0,train_positive_word_probablity,train_negative_word_probablity,vocabulary,newvectorizer)
    valid_accuracy=calAccuracy(valid_predict_Y,valid_y)
    return valid_accuracy

# this vectorizer will skip stop words
vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features=2000
)

# fit the vectorizer on the text
words=vectorizer.fit(imdb_data['review'])

# get the vocabulary

inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
defaultWordDict={key: 0 for key in vocabulary}


train_p1,train_p0,train_positive_Xarr,train_negative_Xarr=getPy(train_x,train_y)




pool = mp.Pool(mp.cpu_count())

print("counting wrods")
train_positive_word_probablity=countWords(train_positive_Xarr,1,vocabulary,defaultWordDict,vectorizer)
train_negative_word_probablity=countWords(train_negative_Xarr,1,vocabulary,defaultWordDict,vectorizer)


print("predicting the validx")

valid_predict_Y=predictY(valid_x,train_p1,train_p0,train_positive_word_probablity,train_negative_word_probablity,vocabulary,vectorizer)

valid_accuracy=calAccuracy(valid_predict_Y,valid_y)

print("Step3: validation accuracy is "+str(valid_accuracy))
print("Step3:output to test-prediction1.csv")
test_predict_Y=predictY(test_x,train_p1,train_p0,train_positive_word_probablity,train_negative_word_probablity,vocabulary,vectorizer)

test_Y_dataframe=DataFrame(test_predict_Y,columns=['PredictY'])
test_Y_dataframe.to_csv('test-prediction1.csv',index=False)


print("Step4:Tuning smoothing parameter alpha")

alphaTask=[0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
result=pool.map(tuneAplpha,alphaTask)
print(alphaTask)
print(result)

bestAccuracy=max(result)
bestAlpha=alphaTask[result.index(bestAccuracy)]

print("The best alpha is "+str(bestAlpha)+" with the accuracy of "+str(bestAccuracy))


plt.figure()
plt.plot(alphaTask, result) 
  
# naming the x axis 
plt.xlabel('Alpha') 
# naming the y axis 
plt.ylabel('Accuracy') 
plt.savefig('Train_Accuracy_Alpha.png')
print("Step4:save the result to Train_Accuracy_Alpha.png")

print("Step4:output to test-prediction2.csv")
train_positive_word_probablity=countWords(train_positive_Xarr,bestAlpha,vocabulary,defaultWordDict,vectorizer)
train_negative_word_probablity=countWords(train_negative_Xarr,bestAlpha,vocabulary,defaultWordDict,vectorizer)
test_predict_Y2=predictY(test_x,train_p1,train_p0,train_positive_word_probablity,train_negative_word_probablity,vocabulary,vectorizer)
test_Y_dataframe=DataFrame(test_predict_Y2,columns=['PredictY'])
test_Y_dataframe.to_csv('test-prediction2.csv',index=False)



# print("Step5:Tune max_features")

# featuresTask=[i for i in range(5000,16000,1000)]
# result=pool.map(tuneMaxFeatures,featuresTask)
# print(featuresTask)
# print(result)


# print("Step5:Tune max_features")

# featuresTask=[i for i in range(100,1000,100)]
# result=pool.map(tuneMaxFeatures,featuresTask)
# print(featuresTask)
# print(result)

print("Step5:Tune both")

alphaTask=[0.2,0.4,0.6,0.8,1]
# featuresTask=[700]*len(alphaTask)
featuresTask=[100,300,700,900]
bothtasks=[(x,y) for x in featuresTask for y in alphaTask]


result=pool.starmap(tuneBoth,bothtasks)
# print(bothtasks)
# print(result)

bestAccuracy=max(result)
bestcombo=bothtasks[result.index(bestAccuracy)]

print(bestcombo)
print(bestAccuracy)

pool.close()

newvectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features=bestcombo[0],
)
# fit the vectorizer on the text
words=newvectorizer.fit(imdb_data['review'])
# get the vocabulary
inv_vocab = {v: k for k, v in newvectorizer.vocabulary_.items()}
newvoca = [inv_vocab[i] for i in range(len(inv_vocab))]
newdefaultWordDict={key: 0 for key in newvoca}
print(len(newvoca))
    
print("Step5:output to test-prediction3.csv")
train_positive_word_probablity=countWords(train_positive_Xarr,bestcombo[1],newvoca,newdefaultWordDict,newvectorizer)
train_negative_word_probablity=countWords(train_negative_Xarr,bestcombo[1],newvoca,newdefaultWordDict,newvectorizer)
test_predict_Y3=predictY(test_x,train_p1,train_p0,train_positive_word_probablity,train_negative_word_probablity,newvoca,newvectorizer)
test_Y_dataframe=DataFrame(test_predict_Y3,columns=['PredictY'])
test_Y_dataframe.to_csv('test-prediction3.csv',index=False)
    
