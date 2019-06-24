from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
CORS(app)

labels = np.load('labels.npy', allow_pickle=True).item()
model = tf.keras.models.load_model('model.h5')
vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore('vocab')

@app.route('/')
def returnRoute():
    return """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>Document Classifier</title>
    <style>
    html, body {
      margin: 0px;
      padding: 0px;
      width: 100%;
      height: 100%;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    h1, #description, #submit-container {
      display: flex;
      justify-content: center;
      align-items: center;
    }
    h1 {
      width: 100%;
      height: 50px;
    }
    #description {
      width: 100%;
      height: 37.5px;
    }
    #submit-container {
      width: 100%;
      flex-wrap: wrap;
    }
    #document-input {
      width: 100%;
    }
    #submit-button {
      margin: 10px;
    }
    </style>
    <script>
    const postDocument = () => {
      const doc = document.getElementById('document-input').value
      fetch('/prediction', {
        method: 'POST',
        cache: 'no-cache',
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
        body: JSON.stringify({ 'prediction': doc })
      })
        .then(response => response.json())
        .then(json => { console.log(json) })
        .catch(error => { console.log(error) })
    }
    </script>
  </head>
  <body>
    <div id="app">
      <h1>Text Taxonomy</h1>
      <div id="description">
        This is a cool application to classify certain documents based on the hash values of the words that compose them.
      </div>
      <div id="submit-container">
        <textarea id="document-input" rows="5"></textarea>
        <button id="submit-button" onclick="postDocument()">
          Submit
        </button>
      </div>
    </div>
  </body>
</html>
    """

@app.route('/prediction', methods=['POST'])
def getPostedPrediction():
    predictionObj = request.get_json()
    predictedClass, predictedProbability = runPrediction(predictionObj['prediction'])
    return jsonify({'Class': str(predictedClass), 'Probability': str(predictedProbability)})

def runPrediction(document):

    x = pd.Series([document])
    xTransformed = vocabProcessor.fit_transform(x)
    xs = np.array(list(xTransformed), dtype=np.int64)

    predictions = model.predict(xs)
    predicted = predictions[0]

    maxVal = -1
    predClass = -1
    predProbability = 0
    for i in range(len(predicted)):
        pred = predicted[i]
        if (pred > maxVal):
            maxVal = pred
            predClass = i
            predProbability = pred

    for key, val in labels.items():
        if predClass == val:
            pred = key
    
    return pred, predProbability

def getApp():
    return app

def main():
    pred, probability = runPrediction('')
    print(pred)
    print(probability)

if __name__ == '__main__':
    app.run()
    

