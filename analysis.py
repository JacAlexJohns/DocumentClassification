import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def main():
    data = readDataFromFile('shuffled-full-set-hashed.csv')
    average = getAverageDocumentLength(data)
    plotDocumentLengths(data, average)
    plotDocumentLengthsPerClass('labels.npy', data)
    plotNumberOfDocumentsPerClass('labels.npy', data)

if __name__ == '__main__':
    main()