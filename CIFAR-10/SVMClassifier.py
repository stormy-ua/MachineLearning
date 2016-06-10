class SVMClassifier:
    losses = []
    batchLosses = []
    
    def __init__(self, epochs=300, learningRate=0.00005, batchSize=250):
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        
    def forwardPropagation(self, W, X, vectY):
        # 1
        self.f1 = W.dot(X.T)
        # 2
        self.f2 = self.f1 * vectY
        #3
        self.f3 = np.sum(self.f2, axis=0)
        #4
        self.f4 = self.f1 - self.f3 + 1
        #5
        self.f5 = self.f4*(1-vectY)
        #6
        self.f6 = np.maximum(self.f5, 0)
        #7
        self.f7 = np.sum(self.f6)
        #8
        self.f8 = self.f7/len(X)
        
        loss = np.sum(self.f8)
        return loss
    
    def backPropagation(self, X, vectY):
        #8
        df7 = 1/len(X)
        #7
        #df7 = np.ones(shape = self.f7.shape)
        df6 = np.ones(shape=self.f6.shape) * df7
        #6
        #df6 = np.ones(shape = self.f6.shape)
        df5 = np.array(self.f5 > 0, dtype = np.float32) * df6
        #5
        #df5 = np.ones(shape = self.f5.shape)
        df4 = df5*(1 - vectY)
        #4
        #df4 = np.ones(shape = self.f4.shape)
        df3 = -1*np.ones(shape=self.f3.shape)*np.sum(df4, axis = 0)
        #3+2
        df1 = df4
        df2 = df3
        df1 += df2*vectY
        #1
        #df1 = np.ones(shape = self.f1.shape)*(1-vectY)
        dW = df1.dot(X)
        
        
        #df4 = np.ones(shape = self.f4.shape)
        #df3 = -self.f1.shape[0]*df4
        #df1 = df4
        #df2 = np.ones(shape = self.f2.shape)*df3
        #df1 += vectY*df2
        #dW = df1.dot(X)
        
        grad = dW
        return grad
    
    def fit(self, X, y):
        self.losses = []
        self.batchLosses = []
        self.labelsCardinality = len(np.unique(y))
        self.W = np.random.randn(self.labelsCardinality, X.shape[1])
        vectY = np.zeros(shape=(self.labelsCardinality, X.shape[0]))
        for i in range(0, X.shape[0]):
            vectY[y[i], i] = 1

        bar = pyprind.ProgBar(self.epochs*len(X)/self.batchSize, bar_char='█', width=100)
        for i in range(0, self.epochs):
            loss = self.forwardPropagation(self.W, X, vectY)
            self.losses.append(loss)
            #logging.info(loss)
            indexes = np.arange(0, len(X))
            np.random.shuffle(indexes)
            trainX = X[indexes]
            trainY = y[indexes]
            for batch in range(0, len(X), self.batchSize):
                batchX = trainX[batch:batch+self.batchSize]
                batchY = trainY[batch:batch+self.batchSize]
                batchVectY = np.zeros(shape=(self.labelsCardinality, batchX.shape[0]))
                for i in range(0, batchX.shape[0]):
                    batchVectY[batchY[i], i] = 1
                batchLoss = self.forwardPropagation(self.W, batchX, batchVectY) 
                self.batchLosses.append(batchLoss)
                grad = self.backPropagation(batchX, batchVectY)
                self.W += -self.learningRate*grad
                bar.update()
    
    def predict(self, X):
        return np.argmax(self.W.dot(X.T), axis=0)