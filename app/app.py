import pickle

import numpy as np
import sklearn
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def home():
    return "<p>Hello World</p>"


@app.route('/predict/', methods=['POST'])
def predict():
    request_data = request.get_json()['body']
    return jsonify(predictor(request_data))


def predictor(data=[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]):
    filename = './app/models/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    prediction = loaded_model.predict([data])
    prediction = int(prediction[0])
    return prediction
