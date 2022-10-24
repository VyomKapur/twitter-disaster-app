import pickle
import pandas as pd
import numpy as np
import re
import twint
import nest_asyncio

from flask import Flask, redirect, url_for, render_template, request, flash

nest_asyncio.apply()
app = Flask(__name__)

model = pickle.load(open('./models/model.pkl', 'rb'))
vec = pickle.load(open('./models/vec.pkl', 'rb'))

def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]', '', text)
    text = re.sub(r'#[A-Za-z0-9]', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text


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
    if request.method == 'POST':
        limit = request.form['limit']
        c = twint.Config()
        c.Search = "disaster"
        c.Store_json = True
        c.Output = "./datasets/dataset.json"
        c.Limit = limit
        twint.run.Search(c)
        data = pd.read_json('./datasets/dataset.json', lines=True)
    return clean_data(data)

def clean_data(data):
    data = data[['tweet', 'language']]
    data = data[data['language'] == 'en']
    data.drop_duplicates(subset=['tweet'], inplace=True)
    original_data = data['tweet']
    data['tweet'] = data['tweet'].apply(cleanText)
    return predict(list(data['tweet']), list(original_data))

@app.route('/predict')
def predict(data, orginal_data):
    data = np.array(data)
    x = vec.transform(data)
    output = model.predict(x)
    return render_template('result.html', pred_text = orginal_data, pred = output)

if __name__ == "__main__":
    app.run(debug=True)