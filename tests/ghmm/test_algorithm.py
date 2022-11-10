import unittest
import numpy as np
import numpy.typing as npt
from scipy.stats import norm

import ghmm.algorithm as algo

class test_Alogirthm(unittest.TestCase):
    tolerance = 1e-10

    O = np.array(range(4))
    A = np.ones((3, 3)) / 9
    mu = np.array(range(3))
    sigma = np.array(range(1, 4))
    pi = np.ones((3)) / 3

    def SMAPE(self, M1: npt.NDArray, M2: npt.NDArray):
        smape = (np.absolute(M1 - M2) / (np.absolute(M1) + np.absolute(M2)).mean()).mean() * 1e+2
        return smape

    def Frames(self, O: npt.NDArray, mu: npt.NDArray, simga: npt.NDArray):
        (T, ) = O.shape
        (N, ) = mu.shape

        frames = np.empty((T, N))
        for t in range(T):
            for i in range(N):
                frames[t, i] = norm.pdf(O[t], mu[i], simga[i])

        return frames

    def test_Frames(self):
        smape = self.SMAPE(self.Frames(self.O, self.mu, self.sigma), algo.Frames(self.O, self.mu, self.sigma))
        self.assertLessEqual(smape, self.tolerance)
    
    def Forward(self, A: npt.NDArray, frames: npt.NDArray, pi: npt.NDArray):
        (T, N) = frames.shape
        alpha = np.empty((T, N))
        for i in range(N):
            alpha[0, i] = pi[i] * frames[0, i]
        for t in range(1, T):
            for j in range(N):
                x = 0 
                for i in range(N):
                   x = x + alpha[t-1, i] * A[i, j]
                alpha[t, j] = x * frames[t, j]

        return alpha

    def test_Forward(self):
        frames = self.Frames(self.O, self.mu, self.sigma)

        smape = self.SMAPE(self.Forward(self.A, frames, self.pi), algo.Forward(self.A, frames, self.pi))
        self.assertLessEqual(smape, self.tolerance)
