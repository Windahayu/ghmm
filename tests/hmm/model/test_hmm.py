import unittest
import numpy as np
from utils import SMAPE
from hmm.model.hmm import Model, BaseModel

class HMM(unittest.TestCase):
    tolerance = 1e-10

    T = 4
    N = 3
    K = 5
    O = np.array(range(T), dtype=np.float_)
    V = np.array(range(K))
    A = np.ones((N, N), dtype=np.float_) / N
    B = np.ones((N, K), dtype=np.float_) / N
    startprobs = np.ones((N), dtype=np.float_) / N
    tol = np.float_(1e-5)
    niter = np.int_(100)

    def model(self):
        testmodel = BaseModel(self.A, self.V, self.B, self.startprobs)
        model = Model(self.A, self.V, self.B, self.startprobs)
        return(testmodel, model)

    def test_Frames(self):
        (testmodel, model) = self.model()
        args = (self.O, self.V, self.B)
        smape = SMAPE(testmodel.Frames(*args), model.Frames(*args))
        self.assertLessEqual(smape, self.tolerance)

    def test_Forward(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.V, self.B)
        args = (self.A, frames, self.startprobs)
        smape = SMAPE(testmodel.Forward(*args), model.Forward(*args))
        self.assertLessEqual(smape, self.tolerance)

    def test_Backward(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.V, self.B)
        args = (self.A, frames)
        smape = SMAPE(testmodel.Backward(*args), model.Backward(*args))
        self.assertLessEqual(smape, self.tolerance)

    def test_Likelihood(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.V, self.B)
        alpha = testmodel.Forward(self.A, frames, self.startprobs)
        args = (alpha, )
        smape = SMAPE(np.array([testmodel.Likelihood(*args)]), np.array([model.Likelihood(*args)]))
        self.assertLessEqual(smape, self.tolerance)

    def test_Gamma(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.V, self.B)
        alpha = testmodel.Forward(self.A, frames, self.startprobs)
        beta = testmodel.Backward(self.A, frames)
        args = (alpha, beta)
        smape = SMAPE(testmodel.Gamma(*args), model.Gamma(*args))
        self.assertLessEqual(smape, self.tolerance)

    def test_Xi(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.V, self.B)
        alpha = testmodel.Forward(self.A, frames, self.startprobs)
        beta = testmodel.Backward(self.A, frames)
        args = (self.A, frames, alpha, beta)
        smape = SMAPE(testmodel.Xi(*args), model.Xi(*args))
        self.assertLessEqual(smape, self.tolerance)

    def test_RestimateA(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.V, self.B)
        alpha = testmodel.Forward(self.A, frames, self.startprobs)
        beta = testmodel.Backward(self.A, frames)
        xi = testmodel.Xi(self.A, frames, alpha, beta)
        gamma = testmodel.Gamma(alpha, beta)
        args = (xi, gamma)
        model.RestimateA(*args)
        testmodel.RestimateA(*args)
        smape = SMAPE(testmodel.A, model.A)
        self.assertLessEqual(smape, self.tolerance)
    
    def test_RestimateB(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.V, self.B)
        alpha = testmodel.Forward(self.A, frames, self.startprobs)
        beta = testmodel.Backward(self.A, frames)
        gamma = testmodel.Gamma(alpha, beta)
        args = (self.O, gamma)
        model.RestimateB(*args)
        testmodel.RestimateB(*args)
        smape = SMAPE(testmodel.B, model.B)
        self.assertLessEqual(smape, self.tolerance)

    def test_RestimateStartProbs(self):
        (testmodel, model) = self.model()
        frames = testmodel.Frames(self.O, self.V, self.B)
        alpha = testmodel.Forward(self.A, frames, self.startprobs)
        beta = testmodel.Backward(self.A, frames)
        gamma = testmodel.Gamma(alpha, beta)
        args = (gamma, )
        model.RestimateStartProbs(*args)
        testmodel.RestimateStartProbs(*args)
        smape = SMAPE(testmodel.startprobs, model.startprobs)
        self.assertLessEqual(smape, self.tolerance)

    def test_Fit(self):
        (testmodel, model) = self.model()
        args = (self.O, self.tol, self.niter)
        model.Fit(*args)
        testmodel.Fit(*args)
        smapeA = SMAPE(testmodel.A, model.A)
        self.assertLessEqual(smapeA, self.tolerance)
        smapeB = SMAPE(testmodel.B, model.B)
        self.assertLessEqual(smapeB, self.tolerance)
        smapestartprobs = SMAPE(testmodel.startprobs, model.startprobs)
        self.assertLessEqual(smapestartprobs, self.tolerance)

