from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators


from sklearn.externals import joblib
import sqlite3
import os
import io
from goose import Goose
import pandas as pd
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

#cur_dir = os.path.dirname(__file__)


category_list = ['televisores', 'celulares', 'geladeira', 'ar_condicionado', 'fogao', 'games', 'notebooks', 'lavadora']
stopwords = stopwords.words('portuguese')
st = SnowballStemmer('portuguese')
goose = Goose()

def preprocess(title_list):
    list_lines = []
    for line in title_list:
        list_lines.append(word_tokenize(unicode(line)))

    #removing stop_words
    list_lines = list_lines
    
    filtered_list = []
    for line in list_lines:
        filtered_words = []
        for word in line:
            if word not in stopwords:
                filtered_words.append(word)
        filtered_list.append(filtered_words)

    #stemming
    
    stemmed_list = []
    for line in filtered_list:
        filtered_words = []
        for word in line:
            filtered_words.append(st.stem(word))
        stemmed_list.append(filtered_words)
    
    preprocessed_list = []
    for line in stemmed_list:
        preprocessed_list.append(' '.join(line))
    
    return preprocessed_list    

def extract(url):
    
    article = goose.extract(url)
    title = article.title
    title = preprocess([title])
    return title

def vectorize(title):
    vectorizer = joblib.load(io.open(os.path.join("C:\\Users\\hsantana\\OneDrive - Braspag Tecnologia em Pagamento Ltda\\Jupyter\\API product classifier", 'pkl_objects', 'vectorizer.pkl'), 'rb'))
    vectorized_title  = vectorizer.transform(title)
    return vectorized_title

def classify(title):
    clf = joblib.load(io.open(os.path.join("C:\\Users\\hsantana\\OneDrive - Braspag Tecnologia em Pagamento Ltda\\Jupyter\\API product classifier", 'pkl_objects', 'clf.pkl'), 'rb'))
    prediction = clf.predict(title)[0]
    for l, item in enumerate(category_list) :
        if prediction == l:
            predict_category = item
    return predict_category

class URLForm(Form):
    urlinput = TextAreaField('', [validators.DataRequired(), validators.length(min=15)])

@app.route('/')
def index():
    form = URLForm(request.form)
    return render_template('URLform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = URLForm(request.form)
    if request.method == 'POST' and form.validate():
        new_input = request.form['urlinput']
        title = extract(new_input)
        vectorized_title = vectorize(title)
        prediction = classify(vectorized_title)
        product = goose.extract(new_input).title
        return render_template('results.html', content=new_input, prediction=prediction, product= product)
    return render_template('URLform.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
