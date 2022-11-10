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

    def Backward(self, A: npt.NDArray, frames: npt.NDArray):
        (T, N) = frames.shape

        beta = np.empty((T, N))
        for i in range(N):
            beta[T - 1, i] = 1

        for t in range(T - 2, -1, -1):
            for i in range(N):
                x = 0
                for j in range(N):
                    x = x + A[i, j] * frames[t + 1, j] * beta[t + 1, j]
                beta[t, i] = x

        return beta

    def test_Backward(self):
        frames = self.Frames(self.O, self.mu, self.sigma)
        
        smape = self.SMAPE(self.Backward(self.A, frames), algo.Backward(self.A, frames))
        self.assertLessEqual(smape, self.tolerance)

    def Likelihood(self, alpha: npt.NDArray):
        (T, N) = alpha.shape
        
        L = 0
        for i in range(N):
            L = L + alpha[T - 1, i]

        return L

    def test_Likelihood(self):
        frames = self.Frames(self.O, self.mu, self.sigma)
        alpha = self.Forward(self.A, frames, self.pi)

        smape = self.SMAPE(np.array([self.Likelihood(alpha)]), np.array([algo.Likelihood(alpha)]))
        self.assertLessEqual(smape, self.tolerance)

    def Xi(self, A: npt.NDArray, frames: npt.NDArray, alpha: npt.NDArray, beta: npt.NDArray):
        (T, N) = frames.shape

        L = self.Likelihood(alpha)

        xi = np.empty((T-1, N, N))
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * frames[t+1, j] * beta[t+1, j] / L
        
        return xi

    def test_Xi(self):
        frames = self.Frames(self.O, self.mu, self.sigma)
        alpha = self.Forward(self.A, frames, self.pi)
        beta = self.Backward(self.A, frames)

        smape = self.SMAPE(self.Xi(self.A, frames, alpha, beta), algo.Xi(self.A, frames, alpha, beta))
        self.assertLessEqual(smape, self.tolerance)
