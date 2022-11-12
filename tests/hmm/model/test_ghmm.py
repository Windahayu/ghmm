import unittest
import numpy as np
from utils import SMAPE
from hmm.model.ghmm import Model, BaseModel

class GHMM(unittest.TestCase):
    tolerance = 1e-10

    T = 4
    N = 3
    O = np.array(range(T), dtype=np.float_)
    A = np.ones((N, N), dtype=np.float_) / N
    means = np.array(range(N), dtype=np.float_)
    variances = np.array(range(1, N+1), dtype=np.float_)
    startprobs = np.ones((N), dtype=np.float_) / N
    min_variance = np.float_(1e-2)

    def model(self):
        testmodel = BaseModel(self.A, self.means, self.variances, self.startprobs, self.min_variance)
        model = Model(self.A, self.means, self.variances, self.startprobs, self.min_variance)
        return(testmodel, model)

    def test_Frames(self):
        (testmodel, model) = self.model()
        args = (self.O, self.means, self.variances)
        smape = SMAPE(testmodel.Frames(*args), model.Frames(*args))
        self.assertLessEqual(smape, self.tolerance)
    
    def test_RestimateB(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.means, self.variances)
        alpha = testmodel.Forward(self.A, frames, self.startprobs)
        beta = testmodel.Backward(self.A, frames)
        gamma = testmodel.Gamma(alpha, beta)
        args = (self.O, gamma)
        model.RestimateB(*args)
        testmodel.RestimateB(*args)
        smapemeans = SMAPE(testmodel.means, model.means)
        self.assertLessEqual(smapemeans, self.tolerance)
        smapevariances = SMAPE(testmodel.variances, model.variances)
        self.assertLessEqual(smapevariances, self.tolerance)
