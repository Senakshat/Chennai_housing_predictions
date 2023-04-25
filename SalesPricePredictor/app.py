# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 23:44:35 2023

@author: aksha
"""

import numpy as np
from flask import Flask,request,render_template
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
    return render_template('style.css')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x)for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    output=round(prediction[0],2)
    return render_template('index.html', prediction_text='Predicted house pricing is Rs.{}'.format(output))
if __name__=="__main__":
    app.run()