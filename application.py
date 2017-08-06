import json
from sklearn.externals import joblib
import numpy as np
import collections
import traceback
import os
import re
from flask import Flask, request

application = Flask(__name__)
count_vect = None
tf_transformer = None
model = None
tags = ['sports', 'entertainment', 'technology', 'politics']

def initialize():
    global count_vect, tf_transformer, model
    count_vect = joblib.load('./model/count_vect.pkl')
    tf_transformer = joblib.load('./model/tf_transform.pkl')
    model = joblib.load('./model/model.pkl')

def cleanString(text):
    t = re.sub('\W+', ' ', text)
    t = ' '.join([s for s in t.split() if not s.isdigit()])
    return t

def authenticate_user(key):
    return True

@application.route('/')
def landing_page():
    return '<h1 align="center">Welcome to document class perfiction API.</h1>'

@application.route('/getClass', methods=['GET', 'POST'])
def get_class():
    global model, count_vect, tf_transformer
    if request.method == 'POST' or request.method == 'GET':
        key = ''
        text = ''
        if 'key' in request.args:
            key = request.args.get('key')
        if 'text' in request.args:
            text = request.args.get('text')
        if authenticate_user(key) and text != '':
            #predict class
            t = []
            t.append(cleanString(text))
            x_test_counts = count_vect.transform(t)
            x_test_tf = tf_transformer.transform(x_test_counts)
            predicted = model.predict(x_test_tf)
            predicted_class = tags[predicted[0]]
            output = {}
            output['status'] = 200
            output['prediction'] = predicted_class
            return json.dumps(output)
        else:
            #Error request params not correct
            output = {}
            output['status'] = 400
            output['prediction'] = "Bad request"
            return json.dumps(output)
    else:
        #Error not a post request
        output = {}
        output['status'] = 400
        output['prediction'] = "Bad request"
        return json.dumps(output)

if __name__ == '__main__':
    initialize()
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)))