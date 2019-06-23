import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_transform as tft

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras import optimizers

def readDataFromFile(file):
    data = pd.read_csv(file).values
    return data

def saveVocabulary(vocabProcessor):

    vocabProcessor.save('vocab')
    vocabDict = vocabProcessor.vocabulary_._mapping

    with open('vocab.txt', 'w') as f:
        for key, val in vocabDict.items():
            f.write(str(key) + ':' + str(val) + '\n')


def saveLabels(labels):
    np.save('labels.npy', labels)

def splitData(data, trainPercent, documentLength):

    np.random.shuffle(data)

    ys = []
    xs = []
    for d in data:
        ys.append(d[0])
        if type(d[1]) != float:
            xs.append(d[1])
        else:
            xs.append('')

    labels = {}
    yVals = list(set(ys))
    for i in range(len(yVals)):
        labels[yVals[i]] = i
    saveLabels(labels)
    ys = np.array([labels[y] for y in ys], dtype=np.int32)

    vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor(documentLength)
    xTransformed = vocabProcessor.fit_transform(xs)
    saveVocabulary(vocabProcessor)

    xs = np.array(list(xTransformed), dtype=np.int64)
    numberOfWords = len(vocabProcessor.vocabulary_)

    split = int(trainPercent * len(ys))

    yTrain = ys[:split]
    xTrain = xs[:split]

    yTest = ys[split:]
    xTest = xs[split:]

    print(max([max(x) for x in xTrain]))

    return xTrain, yTrain, xTest, yTest, len(yTrain), len(yTest), numberOfWords, labels

def rnnModel(batchSize, numberOfWords, documentLength, numberOfClasses, embeddingSize, learnRate):

    model = Sequential()
    model.add(Embedding(numberOfWords, embeddingSize, input_length=documentLength))
    model.add(GRU(embeddingSize))
    model.add(Dense(numberOfClasses, activation='softmax'))
    adam = optimizers.Adam(lr=learnRate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def train(data, trainPercent=.8, documentLength=1000, steps=200, embeddingSize=10, learnRate=0.01, batchNumber=1):

    xTrain, yTrain, xTest, yTest, batchSize, testBatchSize, numberOfWords, labels = splitData(data, trainPercent, documentLength)

    batchSize = max(int(batchSize / batchNumber), 1)

    model = rnnModel(batchSize, numberOfWords, documentLength, len(labels), embeddingSize, learnRate)
    model.summary()
    model.fit(xTrain, yTrain, batchSize, epochs=steps, validation_data=(xTest, yTest))

    model.save('model.h5')

    if (testBatchSize > 0):

        score, acc = model.evaluate(xTest, yTest, batch_size=testBatchSize)

        print('Test Score:', score, '\nTest Accuracy:', acc)

def main():
    file = 'shuffled-full-set-hashed.csv'
    data = readDataFromFile(file)
    train(data, trainPercent=.8, documentLength=300, steps=75, embeddingSize=20, learnRate=0.025, batchNumber=1)

if __name__ == '__main__':
    main()