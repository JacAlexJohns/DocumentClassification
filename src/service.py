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
      background-color: rgba(6, 56, 82, .5);
      background-image: linear-gradient(10deg, rgba(240, 129, 15, .5) 50%, transparent 50%), linear-gradient(-60deg, rgba(230, 223, 68, .5) 30%, transparent 30%);
    }
    h1, h2, #description, #submit-container, #button-container {
      display: flex;
      justify-content: center;
      align-items: center;
    }
    h1, h2 {
      width: 100%;
      height: 50px;
    }
    ul {
      margin: 0 50px;
    }
    #description {
      width: 100%;
      height: 37.5px;
    }
    #submit-container {
      width: 100%;
      flex-wrap: wrap;
    }
    #document-input, #button-container {
      width: 100%;
      margin: 0 50px;
    }
    #submit-button {
      width: 75px;
      height: 25px;
      border-radius: 5px;
      color: white;
      background-color: rgba(6, 56, 82, .8);
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
        .then(json => {
          console.log(json)
          document.getElementById('class-label').innerHTML = 'Class: ' + json['Class']
          document.getElementById('probability-label').innerHTML = 'Probability: ' + json['Probability']
        })
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
      <h2>Endpoints</h2>
      <ul>
        <li><pre><b>/</b>            (current page)</pre></li>
        <li><pre><b>/prediction</b>  POST endpoint: Requires document to be passed as value to key 'prediction'</pre></li>
      </ul>
      <h2>Submit a Document</h2>
      <div id="submit-container">
        <textarea id="document-input" rows="5"></textarea>
        <div id="button-container">
          <button id="submit-button" onclick="postDocument()">
            <b>Submit</b>
          </button>
        </div>
        <div id="class-label"/>
        <div id="probability-label"/>
      </div>
      <h2>Some Fun Images!</h2>
      <img src="../analysisPlots/document-lengths-labeled.png">
      <img src="../analysisPlots/documents-per-class.png">
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
