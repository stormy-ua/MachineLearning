import pickle
import os
import numpy as np

def unpickleCifar(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def load():
    trainX = np.array([], dtype=np.uint8)
    trainX.shape=(0, 3072)
    trainY = np.array([], dtype=np.int32)
    for i in range(1, 6):
        batch = unpickleCifar(os.path.dirname(__file__) + '/cifar-10/data_batch_%d'%i)
        trainX = np.concatenate((trainX, batch[b'data']), axis = 0)
        trainY = np.concatenate((trainY, batch[b'labels']), axis = 0)
    testBatch = unpickleCifar(os.path.dirname(__file__) + '/cifar-10/test_batch')
    testX = testBatch[b'data']
    testY = testBatch[b'labels']
    oneHotTrainY = np.zeros(shape=(len(np.unique(trainY)), trainX.shape[0]))
    for i in range(0, trainX.shape[0]):
        oneHotTrainY[trainY[i], i] = 1
    oneHotTestY = np.zeros(shape=(len(np.unique(trainY)), testX.shape[0]))
    for i in range(0, testX.shape[0]):
        oneHotTestY[testY[i], i] = 1
    batches_meta = unpickleCifar(os.path.dirname(__file__) + '/cifar-10/batches.meta')
    labels = [l.decode() for l in batches_meta[b'label_names']]
    return (trainX, trainY, oneHotTrainY, testX, testY, oneHotTestY, labels)