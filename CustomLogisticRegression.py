import numpy as np
import random
import plotly.express as px



class LogisticRegresssion:


    def __init__(self, epochs=100, learningRate=0.01):
        
        self.features     = None
        self.labels       = None

        self.n            = 0
        self.k            = 0

        self.epochs       = epochs
        self.learningRate = learningRate

        self.weights      = None
        self.bias         = None

        self.errors       = None
    #


    def Sigmoid(self, v):
        
        return (np.exp(v) / (1+np.exp(v)))
    #


    def ScoreThisDP(self, featuresDP):

        return (np.dot(self.weights, featuresDP) + self.bias)
    #


    def ProbThisDP(self, featuresDP):

        return (self.Sigmoid(self.ScoreThisDP(self.weights, self.bias, featuresDP)))
    #


    def LogLossThisDP(self, featuresDP, labelDP):

        probThisDP = self.ProbThisDP(self.weights, self.bias, featuresDP)

        return (-labelDP * np.log(probThisDP)   -   -(1-labelDP) * np.log(1-probThisDP))
    #


    def LogLossAllDP(self):

        result = 0

        for i in range(self.n):

            result += (self.LogLossThisDP(self.weights, self.bias, self.features[i], self.labels[i]))
        #
    #


    def fit(self, X, y):
        
        self.features = X
        self.labels   = y

        self.n        = X.shape[0]
        self.k        = X.shape[1]

        self.epochs       = self.epochs
        self.learningRate = self.learningRate # just for fun, dont force me :)

        self.weights      = np.ones((self.k))
        self.bias         = 0

        self.errors       = np.zeros((self.epochs))



        for epoch in range(self.epochs):

            self.errors[epoch] = self.LogLossAllDP()

            j = random.randint(0, self.n-1)

            guessThisDP = self.ProbThisDP(self.features[j])

            for i in range(self.k):

                self.weights[i] += ((self.labels[j] - guessThisDP) * self.features[i] * self.learningRate)
                self.bias       += ((self.labels[j] - guessThisDP) * self.learningRate)
            #
        #
    #


    def __repr__(self):
        
        errorsPlot = px.scatter(x=range(self.epochs), y=self.weights)
        errorsPlot.show()
    #
#