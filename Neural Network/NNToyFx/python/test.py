import datasets.cifar10 as cifar10
import numpy as np
import NeuralNetwork2Layer

trainX, trainY, vectTrainY, testX, testY, oneHotTestY, labels = cifar10.load()

mean, std = np.mean(trainX), np.std(trainX)
trainX = (trainX - mean)/std
testX = (testX - mean)/std

nn = NeuralNetwork2Layer.NeuralNetworkClassifier(50, 10, epochs=200, learningRate=0.001, l2=0)
nn.fit(trainX, vectTrainY, validationFunc = lambda model: np.sum(model.predict(testX) == testY)/len(testX))