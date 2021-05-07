from flask import Flask, request, render_template, url_for
import pickle
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import io
import re
from sys import path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from string import punctuation, digits
from flask import Flask, request, render_template, url_for
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from langdetect import detect


start_time = time.time()

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST','GET'])
def home():
    #kaydedilen model dosyası alındı
	vec = open("NLP.pkl", 'rb')
	loaded_model = pickle.load(vec)

    #kaydedilen kelime dizisi alındı
	vcb = open("Vocab.pkl", 'rb')
	loaded_vocab = pickle.load(vcb)

	#kullanıcıdan gelen verinin dili bulundu ve eğer ingilizce değilse ingilizceye çevrilerek examples'e eşitlendi
	try:
		examples = request.form['tweet']
		lng=detect(examples) 
		sentence_tr=""
		if lng!="en":
			examples=str(TextBlob(examples).translate(to='en'))


		if lng=="tr":
			sentence_tr=examples
		else:
			sentence_tr=str(TextBlob(examples).translate(to='tr'))
			sentence_tr=re.sub(r"\d+","", sentence_tr)
			sentence_tr=re.sub(r"\r","", sentence_tr)
			sentence_tr=re.sub(r"\t","", sentence_tr)
			sentence_tr=sentence_tr.strip()
	except:
		print("Error")

    #kullanıcıdan gelen veri temizlendi
	examples = examples.lower()
	examples = examples.replace('\n',' ')
	examples = re.sub(r"[^\w\s]", ' ', examples)
	examples = re.sub(r"\d+","", examples)
	examples = re.sub(r"\n"," ", examples)
	examples = re.sub(r"\r","", examples)
	examples = [examples]

    #gelen yorum CountVectorizer ile bir matrixe çevrildi
	count_vect = CountVectorizer(stop_words='english',vocabulary=loaded_vocab)
	x_count = count_vect.fit_transform(examples)

    #model yüklendi ve gelen yorumun sentimenti(duygu analizi tahmin edildi)
	predicted = loaded_model.predict(x_count)

    #tahmin edilen bilgi index.html sayfasına gönderildi.
	result=predicted[0]
	return render_template('index.html',value=result,sentence_tr=sentence_tr,examples_data=request.form['tweet'])