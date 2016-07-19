import numpy as np
import nodes
import pyprind


class NeuralNetworkClassifier:
    losses = []
    batchLosses = []

    def __init__(self, n1, n2, epochs=300, learningRate=0.00005, batchSize=250, l2=0.01):
        self.n1 = n1
        self.n2 = n2
        self.l2 = l2
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.lossNode = nodes.SoftmaxLossNode()
        self.matrixMulNode1 = nodes.MatrixMulNode()
        self.activation1 = self.activationNode()
        self.matrixMulNode2 = nodes.MatrixMulNode()
        self.activation2 = self.activationNode()
        self.dropoutNode = nodes.DropoutNode()

    def activationNode(self):
        return nodes.ReLuNode()

    def forwardPropagation(self, W1, W2, b1, b2, X, vectY):
        # 1
        self.f1 = self.matrixMulNode1.forward(W1, X.T) + b1[:, np.newaxis]
        # 2
        self.f2 = self.activation1.forward(self.f1)
        #self.f25 = self.dropoutNode.forward(self.f2)

        # 3
        self.f3 = self.matrixMulNode2.forward(W2, self.f2) + b2[:, np.newaxis]
        # 4
        self.f4 = self.activation2.forward(self.f3)
        # 5
        self.f5 = self.lossNode.forward(self.f4, vectY)
        self.f6 = self.f5 + self.l2 * (np.sum(W1 ** 2) + np.sum(W1 ** 2) + np.sum(b1 ** 2) + np.sum(b2 ** 2))

        loss = self.f6
        return loss

    def backPropagation(self):
        # 5
        df1 = self.lossNode.backward()
        # 4
        df2 = self.activation2.backward(df1)
        # 3
        (df3, df31) = self.matrixMulNode2.backward(df2)
        # 2
        df4 = self.activation1.backward(df31)
        # 1
        (df5, df51) = self.matrixMulNode1.backward(df4)

        db2 = np.sum(df2, axis=1)
        db1 = np.sum(df4, axis=1)
        grad = (df5 + 2 * self.l2 * self.W1, df3 + 2 * self.l2 * self.W2, db1 + 2 * self.l2 * self.b1,
                db2 + 2 * self.l2 * self.b2)
        return grad

    def fit(self, X, y, validationFunc=None):
        self.losses = []
        self.trainAccuracies = []
        self.validationAccuracies = []
        self.validationAccuracy = None
        self.batchLosses = []
        self.labelsCardinality = y.shape[0]
        self.W1 = 0.01 * np.random.randn(self.n1, X.shape[1])
        self.W2 = 0.01 * np.random.randn(self.n2, self.n1)
        self.b1 = 0.01 * np.ones(self.n1)
        self.b2 = 0.01 * np.ones(self.n2)
        self.mu = 0.9
        self.vW1 = np.zeros(self.W1.shape)
        self.vW2 = np.zeros(self.W2.shape)
        self.vb1 = np.zeros(self.b1.shape)
        self.vb2 = np.zeros(self.b2.shape)
        yr = np.argmax(y, axis=0)
        self.validationAccuracy = 0

        bar = pyprind.ProgBar(self.epochs * len(X) / self.batchSize, bar_char='█', width=60, track_time=True, stream=1)
        for i in range(0, self.epochs):
            loss = self.forwardPropagation(self.W1, self.W2, self.b1, self.b2, X, y)
            self.trainAccuracy = np.sum(self.predict(X) == yr) / len(X)
            if validationFunc != None:
                self.validationAccuracy = validationFunc(self)
                self.validationAccuracies.append(self.validationAccuracy)
            self.losses.append(loss)
            self.trainAccuracies.append(self.trainAccuracy)
            # logging.info(loss)
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            trainX = X[indexes]
            trainY = y[:, indexes]
            for batch in range(0, len(X), self.batchSize):
                batchX = trainX[batch:batch + self.batchSize]
                batchY = trainY[:, batch:batch + self.batchSize]
                batchLoss = self.forwardPropagation(self.W1, self.W2, self.b1, self.b2, batchX, batchY)
                self.batchLosses.append(batchLoss)
                gradW1, gradW2, gradb1, gradb2 = self.backPropagation()
                self.vW1 = self.mu * self.vW1 - self.learningRate * gradW1
                self.vW2 = self.mu * self.vW2 - self.learningRate * gradW2
                self.vb1 = self.mu * self.vb1 - self.learningRate * gradb1
                self.vb2 = self.mu * self.vb2 - self.learningRate * gradb2

                # self.W1 += -self.learningRate*gradW1
                # self.W2 += -self.learningRate*gradW2
                # self.b1 += -self.learningRate*gradb1
                # self.b2 += -self.learningRate*gradb2

                self.W1 += self.vW1
                self.W2 += self.vW2
                self.b1 += self.vb1
                self.b2 += self.vb2
                bar.update(
                    item_id="[%.3f, %.3f] %.2f, %.2f" % (self.trainAccuracy, self.validationAccuracy, loss, batchLoss))
        print(bar)

    def predictOneHot(self, X):
        activation = self.activationNode().forward
        return activation(
            self.W2.dot(self.dropoutNode.p * activation(self.W1.dot(X.T) + self.b1[:, np.newaxis])) + self.b2[:,
                                                                                                      np.newaxis])

    def predict(self, X):
        return np.argmax(self.predictOneHot(X), axis=0)
