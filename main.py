#Importing libraries

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import re
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import string
import pickle
from flask import Flask, render_template, url_for, request, redirect,jsonify
from utils import EncoderLayer,Encoder,PositionalEncoding,MultiHeadAttention


app = Flask(__name__)



#nltk.download('wordnet')
#nltk.download('stopwords')


#Utility functions to help preprocess text and map conntractions
def decontract(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s$", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

stopwords_english = set(stopwords.words('english'))-set(['No','no','not','Not'])

def preprocess(text,stopwords=stopwords_english):
    lemmatizer = WordNetLemmatizer()


    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    #Decontract texts
    text=decontract(text)
    # tokenize texts


    texts_clean = []
    for word in text.split():
        if (word not in stopwords_english and  # remove stopwords
                word not in set(string.punctuation)-set(['!','?','.','@',':'])):  # remove punctuation
            #Lemmatize word 
            lem_word = lemmatizer.lemmatize(word,"v")  # Lemmatizing word
            texts_clean.append(lem_word)

    return " ".join(texts_clean)

# Implemented based on the Attention is all you need paper.
def scaled_dot_product_attention(queries, keys, values, mask):
    # Calculating  QK.T
    product = tf.matmul(queries, keys, transpose_b=True)
    # Get the scale factor
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    # Scale the dot product for improved training speed and stability 
    scaled_product = product / tf.math.sqrt(keys_dim)
    # Apply the mask
    if mask is not None:
        scaled_product += (mask * -1e9)
    # dot product of QK.T with Values 
    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    
    return attention

def create_padding_mask(seq): #seq: (batch_size, seq_length)
# Create the mask for padding
  mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :]

def predict(encoder,tokenized_sentences):
  logits=encoder(tokenized_sentences,create_padding_mask(tokenized_sentences),False)
  predictions=np.argmax(tf.keras.layers.Softmax()(logits),axis=1)
  return predictions

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Encoder':
            return Encoder
        if name== 'PositionalEncoding':
            return PositionalEncoding
        if name=='EncoderLayer':
            return EncoderLayer
        if name=='MultiHeadAttention':
            return MultiHeadAttention
        return super().find_class(module, name)


tokenizer=pickle.load(open('models/tokenizer.sav', 'rb'))
#loaded_model = pickle.load(open('models/finalized_model (1).sav', 'rb'))
loaded_model =CustomUnpickler(open('models/finalized_model (1).sav', 'rb')).load()

print("Tokenizer and Model Loaded")

@app.route('/')
def welcome():
    return render_template('homePage.html')

@app.route('/details', methods = ['POST', 'GET'])
def details():
    inputText = request.form['para']
    sentence=tokenizer.encode(preprocess(inputText))
    sentence=tf.keras.preprocessing.sequence.pad_sequences([sentence],value=0,padding='post',maxlen=50)

    answer=predict(loaded_model,sentence)
    if(answer)==1:
        return render_template('details.html')
    else:
        return render_template('not.html')


if __name__ == "__main__":
    app.run(threaded=True,port=5000)

