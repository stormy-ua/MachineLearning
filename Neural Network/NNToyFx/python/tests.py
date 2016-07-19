import unittest
from nodes import *
from sklearn import datasets
import numpy as np
import NeuralNetwork2Layer
from numpy.testing import *


class TestReLuNode(unittest.TestCase):
    def testForwardPropagationGreaterThan0(self):
        node = ReLuNode()
        self.assertEqual(1, node.forward(1))

    def testForwardPropagationLessThan0(self):
        node = ReLuNode()
        self.assertEqual(0, node.forward(-1))

    def testBackwardPropagationGreaterThan0(self):
        node = ReLuNode()
        node.forward(1)
        self.assertEqual(1, node.backward(1))

    def testBackwardPropagationLessThan0(self):
        node = ReLuNode()
        node.forward(-1)
        self.assertEqual(0, node.backward(1))


class TesrNeuralNetworkClassifier(unittest.TestCase):
    def testShouldOverfitScikitLearnMnistSampleDataset(self):
        mnist = datasets.load_digits()
        trainX = np.array(mnist.data)
        trainY = np.array(mnist.target)
        oneHotTrainY = np.zeros(shape=(len(np.unique(trainY)), trainX.shape[0]))
        for i in range(0, trainX.shape[0]):
            oneHotTrainY[trainY[i], i] = 1
        mean, std = np.mean(trainX), np.std(trainX)
        trainX = (trainX - mean) / std
        nn = NeuralNetwork2Layer.NeuralNetworkClassifier(50, 10, epochs=100, learningRate=0.1, l2=0)
        nn.fit(trainX, oneHotTrainY)
        self.assertEqual(0, np.sum(nn.predict(trainX) != trainY))

