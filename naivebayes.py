"""
Naive Bayes classifier, using numpy

Currently only handles discrete-valued inputs and outputs
"""

from numpy import *

class NB(object):
    def __init__(self, indims, outdim):
        self.indim = len(indims)
        self.max_indims = amax(indims)
        self.feature_count = zeros((outdim, self.indim, self.max_indims))
        self.output_count = zeros((outdim))

    def predict(self, inputs):
        """One line of magic and slicing"""
        return argmax(self.output_count * (self.feature_count[:, arange(self.indim), inputs] / tile(maximum(self.output_count, 1), (self.indim, 1)).T).prod(1))

    def train(self, data, targets):
        i = identity(self.max_indims)
        for x,y in zip(data, targets):
            self.feature_count[y] += i[x]
            self.output_count[y] += 1.0

        error = 0.0
        for x,y in zip(data, targets):
            error += (self.predict(x) - y)**2

        return error / len(data)