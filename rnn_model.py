import numpy as np
import pandas as pd
import tensorflow as tf

def readDataFromFile(file):
    data = pd.read_csv(file).values
    return data

def splitData(data, trainPercent):

    np.random.shuffle(data)

    ys = []
    xs = []
    for d in data:
        if type(d[1]) != float:
            ys.append(d[0])
            xs.append(d[1])

    labels = {}
    yVals = list(set(ys))
    for i in range(len(yVals)):
        labels[yVals[i]] = i
    ys = np.array([labels[y] for y in ys], dtype=np.int32)

    documentLength = 0
    for x in xs:
        documentLength = max(documentLength, len(x.split(' ')))

    vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor(documentLength)
    xTransformed = vocabProcessor.fit_transform(xs)

    xs = np.array(list(xTransformed))
    numberOfWords = len(vocabProcessor.vocabulary_)

    split = int(trainPercent * len(ys))

    yTrain = ys[:split]
    xTrain = xs[:split]

    yTest = ys[split:]
    xTest = xs[split:]

    return xTrain, yTrain, xTest, yTest, len(yTrain), len(yTest), numberOfWords, documentLength, labels

def rnnModel(batchSize, numberOfWords, documentLength, numberOfClasses, embeddingSize, learnRate):

    global trainOp

    tf.compat.v1.reset_default_graph()

    x = tf.compat.v1.placeholder(tf.int32, [None, documentLength], name="Input")
    y = tf.compat.v1.placeholder(tf.int32, [None], name="Label")

    wordVectors = tf.contrib.layers.embed_sequence(x, vocab_size=numberOfWords, embed_dim=embeddingSize)

    wordList = tf.unstack(wordVectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(embeddingSize)

    _, encoding = tf.nn.static_rnn(cell, wordList, dtype=tf.float32)

    logits = tf.layers.dense(encoding, numberOfClasses, activation=None)

    predictedClasses = tf.argmax(logits, 1, name="Output", output_type=tf.int32)

    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learnRate)
    trainOp = optimizer.minimize(loss)

    return {
        'x': x,
        'y': y,
        'probs': logits,
        'pred': predictedClasses,
        'loss': loss,
        'train': trainOp
    }

def train(data, trainPercent=.8, steps=200, embeddingSize=50, learnRate=0.01):

    xTrain, yTrain, xTest, yTest, batchSize, testBatchSize, numberOfWords, documentLength, labels = splitData(data, trainPercent)

    model = rnnModel(batchSize, numberOfWords, documentLength, len(labels), embeddingSize, learnRate)

    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

        feedDict = {
            model['x']: xTrain,
            model['y']: yTrain
        }

        for i in range(steps):
            loss, _ = sess.run([model['loss'], model['train']], feedDict)
            print('Loss at step', i, '=', loss)

        if (testBatchSize > 0):
            feedDict = {
                model['x']: xTest
            }

            predictions = sess.run([model['pred']], feedDict)[0]

            correctCount = sum([int(pred == y) for pred, y in zip(predictions, yTest)])
            accuracy = correctCount / len(yTest)
            print('Accuracy = {0:f}'.format(accuracy))
        
        sess.close()

def main():
    file = 'shuffled-full-set-hashed.csv'
    data = readDataFromFile(file)
    train(data)

if __name__ == '__main__':
    main()