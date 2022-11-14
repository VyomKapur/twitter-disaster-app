import pickle
import pandas as pd
import numpy as np
import re
import twint
import nest_asyncio
import os
from geopy.geocoders import Nominatim
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from flask import Flask,session, render_template, request, session

nest_asyncio.apply()
app = Flask(__name__)
app.secret_key = '\xf0?a\x9a\\\xff\xd4;\x0c\xcbHi'
model_mnb = pickle.load(open('./models/model_mnb.pkl', 'rb'))
model_bnb = pickle.load(open('./models/model_bnb.pkl', 'rb'))
model_svm = pickle.load(open('./models/model_svm.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

vectorizer = pickle.load(open('./models/vec.pkl', 'rb'))
vectorizer1 = pickle.load(open('./models/vec1.pkl', 'rb'))

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

def get_geocode(location):
    if location == "Worldwide":
        return False
    geo = Nominatim(user_agent="Disaster App").geocode(location)
    geocode = str(geo.latitude) + ", " + str(geo.longitude) + ",10km"
    return geocode

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')

@app.route('/contact')
def contactpage():
    return render_template('contact.html')

@app.route('/get-data', methods=['POST', 'GET'])
async def get_data():
    os.remove("./datasets/dataset1.json")
    if request.method == 'POST':
        limit = request.form['limit']
        location = request.form['state']
        geocode = get_geocode(location)
        print(request.form['model'])
        session['model'] = int(request.form['model'])
        c = twint.Config()
        c.Search = "disaster"
        c.Store_json = True
        c.Output = "./datasets/dataset1.json"
        c.Limit = int(limit)
        if geocode != False:
            c.Geo = geocode
        session['limit'] = limit
        session['location'] = location
        twint.run.Search(c)
        data = pd.read_json('./datasets/dataset1.json', lines=True, nrows=int(limit)+2)
    return clean_data(data)

def clean_data(data):
    data = data[['tweet', 'language']]
    data = data[data['language'] == 'en']
    data.drop_duplicates(subset=['tweet'], inplace=True)
    original_data = data['tweet']
    data['tweet'] = data['tweet'].apply(cleanText)
    data['tweet'] = [word_tokenize(entry) for entry in data['tweet']]
    data['tweet'] = data['tweet'].apply(lemmat)
    data['lemmatized_word'] = data['tweet'].apply(lemmat)
    data["lemmatized_word"] = data.lemmatized_word.apply(ListToStr)
    return predict(list(data['lemmatized_word']), list(original_data))

@app.route('/predict')
def predict(data, orginal_data):
    data = np.array(data)
    
    if session['model'] == 0:
        x = vectorizer.transform(data)
        output = model_mnb.predict(x)
        
    elif session['model'] == 1:
        x = vectorizer1.transform(data)
        output = model_bnb.predict(x)

    elif session['model'] == 2:
        x = vectorizer.transform(data)
        output = model_svm.predict(x)

    return render_template('result.html', pred_text = orginal_data, pred = output, limit = session['limit'], location=session['location'])

if __name__ == "__main__":
    app.run(debug=True)