#library used for load data 
import pandas as pd
import numpy as np

#library and packages used for natural language processing 
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#library and packages used for fit the ML model and evaluation  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#Part 1 proccess data input 
def load_data_Sentence(txt_path):
    df = pd.read_csv(txt_path, sep=";", header=None)
    if len(df.columns) == 2:
        df.columns = ["Sentence", "Emotions"]
    else:
        df.columns = ["Sentence"]
    return df['Sentence']

def load_data_Emotions(txt_path):
    df = pd.read_csv(txt_path, sep=";", header=None)
    if len(df.columns) == 2:
        df.columns = ["Sentence", "Emotions"]
    else:
        df.columns = ["Sentence"]
    return df['Emotions']

#Part 2 data preprocessing 
def preprocessing(df):
    stop_words=set(stopwords.words("english"))
    #removeing stopword
    Sentence_Without_Stopword = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])) 
    #Word Tokenization the Sentence 
    Sentence_Tokenize = Sentence_Without_Stopword.apply(word_tokenize)                                                  
    Stemmer = SnowballStemmer("english")
    #Apply stemming to the Sentence
    Sentence_After_Stemming = Sentence_Tokenize.apply(lambda x: [Stemmer.stem(y) for y in x])                           
    lem = WordNetLemmatizer()
    #Apply Lemmatization to the Sentence 
    Sentence_After_Lem = Sentence_After_Stemming.apply(lambda x: [lem.lemmatize(y) for y in x])   
    #merge the words in every row                     
    Sentence_After_Preprocessing = Sentence_After_Lem.apply(lambda x : " ".join(x))
    return Sentence_After_Preprocessing

#part 3 Build ML Model
def Naive_Bayes_classify_Model(X_train, Y_train, X_test):
    cv = CountVectorizer()
    #standardise the data to fit the model 
    # Replace the emotions represetation as number 
    Y_train = Y_train.map({'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5})
    #the words in every sentence become vector
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)
    #Naive Bayes Classifier model
    clf = MultinomialNB().fit(X_train, Y_train)
    #predict and output the result 
    y_pred = clf.predict(X_test)
    return y_pred

#Measure the Mean square error 
def Accuracy_score(Y_test,y_pred):
    y_test = Y_test.map({'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5})
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy_score :', accuracy_score)

# print result 
def result(y_pred):
    df = pd.DataFrame(y_pred, columns = ["Emotions"])
    #replace the the classification form number to emotion words 
    df['Emotions'] = df['Emotions'].replace({0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'})
    return df

#read train dataset
train = 'train.txt'
X_train = load_data_Sentence(train)
Y_train = load_data_Emotions(train)
#preprocess the train dataset
X_train = preprocessing(X_train)

#read val dataset
val = 'val.txt'
X_val = load_data_Sentence(val)
Y_val = load_data_Emotions(val)
#preprocess the val datset
X_val = preprocessing(X_val)

#fit the model by train dataset and use val dataset to evaluate the model 
y_val_pred = Naive_Bayes_classify_Model(X_train, Y_train, X_val)
#Output the mean square error to evalate the model  
Accuracy_score(Y_val,y_val_pred)

#read the test dataset
test = 'test_data.txt'
X_test = load_data_Sentence(test)
#preprocess the test dataset 
X_test = preprocessing(X_test)

#fit the model by train datset and test dataset
y_test_pred = Naive_Bayes_classify_Model(X_train, Y_train, X_test)
#output the result
result = result(y_test_pred)
#Output the result as a txt file 'test_prediction.txt'
result.to_csv('test_prediction.txt', header=None, index = False, sep='\t')