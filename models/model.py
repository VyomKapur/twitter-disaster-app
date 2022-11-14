import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import svm

data = pd.read_csv('./datasets/train.csv')
print(data.head())

lemmatizer = WordNetLemmatizer()

def cleanText(text):
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[^\w\s]','',text)
    text = text.lower()
    return text

def lemmat(text) :
    lem = []
    for word in text :
        lem.append(lemmatizer.lemmatize(word))
    return lem

def ListToStr(l) :
    return ' '.join(l)

vectorizer = CountVectorizer(stop_words='english')
vectorizer1 = CountVectorizer(stop_words='english', binary=True)

y = data['target']

data['text'] = data['text'].apply(cleanText)
data['text'] = [word_tokenize(entry) for entry in data['text']]
data['text'] = data['text'].apply(lemmat)
data['lemmatized_word'] = data['text'].apply(lemmat)
data["lemmatized_word"] = data["lemmatized_word"].apply(ListToStr)

x = vectorizer.fit_transform(data['lemmatized_word'])
x1 = vectorizer1.fit_transform(data['lemmatized_word'])

pickle.dump(vectorizer1, open('./models/vec1.pkl', 'wb'))
pickle.dump(vectorizer, open('./models/vec.pkl', 'wb'))

model_mnb = MultinomialNB()
model_mnb.fit(x,y)

model_bnb= BernoulliNB()
model_bnb.fit(x1,y)

model_svm = svm.SVC(C = 1.0 , kernel = 'linear' , degree = 4 , gamma = 'auto')
model_svm.fit(x,y)

pickle.dump(model_mnb, open('./models/model_mnb.pkl', 'wb'))
pickle.dump(model_bnb, open('./models/model_bnb.pkl', 'wb'))
pickle.dump(model_svm, open('./models/model_svm.pkl', 'wb'))
