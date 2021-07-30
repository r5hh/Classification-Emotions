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
from flask import Flask, request, jsonify, render_template

#Initialize the flask
app = Flask(__name__) 

#Define html file to get user input 
@app.route('/')
def home():
   return render_template('classification.html')

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

#Output reult to page 
@app.route('/predict',methods=['POST'])
def result():
    #read train dataset
    X_train = load_data_Sentence('train.txt')
    Y_train = load_data_Emotions('train.txt')
    #standardize the training dateset 
    X_train = preprocessing(X_train)
    cv = CountVectorizer()
    Y_train = Y_train.map({'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5})
    #fit Naive_Bayes_classification model
    X_train = cv.fit_transform(X_train)
    clf = MultinomialNB().fit(X_train, Y_train)

    if request.method == 'POST':
        #read the input data in webpage 
        User_input = request.form['input']
        #standardize the input data to a pd dataframe
        df = pd.DataFrame({User_input}, columns = ["Sentence"])
        Sentence_input = df['Sentence']
        #perform preprocessing
        Sentence_input = preprocessing(Sentence_input)
        #fit model 
        X_test = cv.transform(Sentence_input)
        result = clf.predict(X_test)
        #output the result 
        if result == 0:
            #represent Anger emoji 
            result = "\U0001F92C"
        elif result == 1:
            #represent Fear emoji
            result ="\U0001F631"
        elif result == 2:
            #represent  Joy emoji
            result ="\U0001F600"
        elif result == 3:
            #represent Love emoji
            result = "\U0001F60D"	
        elif result == 4:
            #represent Sadness emojiv
            result = "\U0001F62D"
        elif result == 5:
            #represent Surprise emoji
            result = "\U0001F632"
            #return result to classification.html
        return render_template("classification.html", result= result)

if __name__ == "__main__":
    app.run(debug=True)