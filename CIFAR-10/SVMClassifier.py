class SVMClassifier:
    losses = []
    
    def __init__(self, epochs=300, learningRate=0.00005, batchSize=250):
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize

    def svmDiff(self, W, X, y):    
        ypred = W.dot(X.T)
        vectY = np.zeros(shape=(self.labelsCardinality, X.shape[0]))
        for i in range(0, X.shape[0]):
            vectY[y[i], i] = 1
        diffs = ypred - np.sum(ypred*vectY, axis=0) + 1
        vectY[vectY == 1] = 2
        vectY[vectY == 0] = 1
        vectY[vectY == 2] = 0
        diffs = diffs*vectY
        return diffs
        
    def svmLoss(self, W, X, y):
        diffs = self.svmDiff(W, X, y)
        loss = np.sum(np.max(diffs, 0))/len(X)
        return loss
    
    def svmGradient(self, idiff, x, y):
        grad = idiff[:, np.newaxis] * x
        grad[y] = -sum(idiff)*x
        return grad

    def svmBatchGradient(self, batchDiff, batchX, batchY):
        batchGrad = np.zeros(shape=(len(labels), batchX.shape[1]))
        size = len(batchX)
        idiff = np.array(batchDiff > 0, dtype=np.int32)
        for i in range(0, size):
            batchGrad += self.svmGradient(idiff[:, i], batchX[i], batchY[i])
        return batchGrad
    
    def fit(self, X, y):
        self.losses = []
        self.labelsCardinality = len(np.unique(y))
        self.W = np.random.randn(self.labelsCardinality, X.shape[1])

        bar = pyprind.ProgBar(self.epochs*len(X)/self.batchSize, bar_char='█', width=100)
        for i in range(0, self.epochs):
            loss = self.svmLoss(self.W, X, y)
            self.losses.append(loss)
            #logging.info(loss)
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            trainX = X[indexes]
            trainY = y[indexes]
            for batch in range(0, len(X), batchSize):
                batchX = trainX[batch:batch+batchSize]
                batchY = trainY[batch:batch+batchSize]
                batchDiff = self.svmDiff(self.W, batchX, batchY)
                grad = self.svmBatchGradient(batchDiff, batchX, batchY)
                self.W += -self.learningRate*grad
                bar.update()
    
    def predict(self, X):
        return np.argmax(self.W.dot(X.T), axis=0)