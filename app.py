from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from keras.models import load_model
from keras import backend
from tensorflow.keras import backend
import tensorflow as tf
global graph
graph=tf.get_default_graph()
from skimage.transform import resize
from flask import Flask,redirect,url_for,request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
app=Flask(__name__)
MODEL_PATH = 'models/braintumor.h5'
model = load_model(MODEL_PATH)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
       f = request.files['file']
       basepath = os.path.dirname(__file__)
       file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
       f.save(file_path)
       img = image.load_img(file_path, target_size=(64,64))
       x = image.img_to_array(img)
       x = np.expand_dims(x,axis=0)
       with graph.as_default():
           preds = model.predict_classes(x)
           index = ['yes','no']
           text = "prediction : "+index[preds[0]]
           return text
       if __name__ == '__main__':
           app.run(debug=False,threaded = False)
"""
Created on Fri Jul 10 11:47:44 2020

@author: Suhasini
"""


