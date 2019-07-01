import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
import json

def readDataFromFile(file):
    data = pd.read_csv(file).values
    return data

def getAverageDocumentLength(documents):
    return sum([len(x.split(' ')) for y, x in documents if type(x) == str]) / len(documents)

def plotDocumentLengths(documents, average):
    xs = []
    for y, x in documents:
        if type(x) == str:
            xs.append(len(x.split(' ')))
        else:
            xs.append(0)
    plt.plot(range(len(xs)), xs, 'bo')
    plt.title('Document Lengths: Average=' + str(average))
    plt.savefig('./analysisPlots/document-lengths.png')
    plt.show()

def plotDocumentLengthsPerClass(labelsFile, documents):
    labels = np.load(labelsFile, allow_pickle=True).item()
    xs = [[] for i in range(len(labels.keys()))]
    ys = [[] for j in range(len(labels.keys()))]
    acc = 0
    for y, x in documents:
        if type(x) == str:
            xs[labels[y]].append(len(x.split(' ')))
        else:
            xs[labels[y]].append(0)
        ys[labels[y]].append(acc)
        acc += 1
    ps = []
    for k in range(len(xs)):
        p = plt.scatter(ys[k], xs[k], marker='o', s=1)
        ps.append(p)
    plt.legend(tuple(ps), tuple(labels.keys()), ncol=4, fontsize=4)
    plt.title('Document Lengths Colored by Class')
    plt.savefig('./analysisPlots/document-lengths-labeled.png')
    plt.show()

def plotNumberOfDocumentsPerClass(labelsFile, documents):
    labels = np.load(labelsFile, allow_pickle=True).item()
    ys = [0 for i in range(len(labels.keys()))]
    for y, x in documents:
        ys[labels[y]] += 1
    plt.bar(labels.keys(), ys)
    plt.xticks(rotation='vertical', fontsize=5)
    plt.subplots_adjust(bottom=0.25)
    plt.title('Number of Documents per Class')
    plt.savefig('./analysisPlots/documents-per-class.png')
    plt.show()

def plotWordFrequenciesFromDocuments(documents):
    print('Starting')
    words = {}
    for y, x in documents:
        if type(x) == str:
            docWords = x.split(' ')
            for word in docWords:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
    print('Processing')
    documentWords = []
    wordTotals = [val for key, val in words.items()]
    average = sum(wordTotals) / len(wordTotals)
    stddev = 0
    for w in wordTotals:
        stddev += (w - average)**2
    stddev /= len(wordTotals)
    stddev = stddev**(1/2)
    wordTotals = []
    for key, val in words.items():
        if (val > (average - stddev) and val < (average + stddev)):
            documentWords.append(key)
            wordTotals.append(val)
    print('Plotting')
    plt.bar(range(len(documentWords)), wordTotals)
    plt.savefig('./analysisPlots/word-frequencies.png')
    plt.show()

def runPrediction(labels, model, vocabProcessor, document):

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

def runPredictionsOnData(data):
    acc = 0
    for d in data:
        label = d[0]
        if (type(d[1]) != str):
            d[1] = ''
        pred, proba = runPrediction(labels, model, vocabProcessor, d[1])
        if pred == label:
            acc += 1
    print(acc)
    print(acc / len(data))

def runAPIPredictionsOnData(data):
    url = 'http://ec2-3-91-87-100.compute-1.amazonaws.com/prediction'
    acc = 0
    for d in data:
        label = d[0]
        if (type(d[1])!= str):
            d[1] = ''
        r = requests.post(url, json={'prediction': d[1]})

        response = r.json()
        pred = response['Class']
        if pred == label:
            acc += 1
        print(pred, ' : ', label, ' : ', acc)

def main():
    # labels = np.load('labels.npy', allow_pickle=True).item()
    # model = tf.keras.models.load_model('model.h5')
    # vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore('vocab')
    data = readDataFromFile('shuffled-full-set-hashed.csv')
    # average = getAverageDocumentLength(data)
    # plotDocumentLengths(data, average)
    # plotDocumentLengthsPerClass('labels.npy', data)
    # plotNumberOfDocumentsPerClass('labels.npy', data)
    # plotWordFrequenciesFromDocuments(data)
    # runPredictionsOnData(data)
    # runAPIPredictionsOnData(data)
    print(56463 / len(data))

        

if __name__ == '__main__':
    main()