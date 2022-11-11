import unittest
import numpy as np
import numpy.typing as npt
from scipy.stats import norm

import ghmm.algorithm as algo

class test_Alogirthm(unittest.TestCase):
    tolerance = 1e-10

    O = np.array(range(4), dtype=np.float_)
    A = np.ones((3, 3), dtype=np.float_) / 9
    mu = np.array(range(3), dtype=np.float_)
    sigma = np.array(range(1, 4), dtype=np.float_)
    pi = np.ones((3), dtype=np.float_) / 3

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

    def Gamma(self, alpha: npt.NDArray, beta: npt.NDArray):
        (T, N) = alpha.shape

        gamma = np.empty((T, N))
        for t in range(T):
            for i in range(N):
                nominator = alpha[t, i] * beta[t, i]

                denominator = 0                
                for j in range(N):
                    denominator = denominator + alpha[t, j] * beta[t, j]

                gamma[t, i] = nominator / denominator

        return gamma


    def test_Gamma(self):
        frames = self.Frames(self.O, self.mu, self.sigma)
        alpha = self.Forward(self.A, frames, self.pi)
        beta = self.Backward(self.A, frames)

        smape = self.SMAPE(self.Gamma(alpha, beta), algo.Gamma(alpha, beta))
        self.assertLessEqual(smape, self.tolerance)

    def Xi(self, A: npt.NDArray, frames: npt.NDArray, alpha: npt.NDArray, beta: npt.NDArray):
        (T, N) = frames.shape

        L = self.Likelihood(alpha)

        xi = np.empty((T-1, N, N))
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    numerator = alpha[t, i] * A[i, j] * frames[t+1, j] * beta[t+1, j]
                    denominator = 0
                    for k in range(N):
                        for l in range(N):
                            denominator = denominator + alpha[t, k] * A[k, l] * frames[t+1, l] * beta[t+1, l]
                    xi[t, i, j] = numerator / denominator
        
        return xi

    def test_Xi(self):
        frames = self.Frames(self.O, self.mu, self.sigma)
        alpha = self.Forward(self.A, frames, self.pi)
        beta = self.Backward(self.A, frames)
        
        smape = self.SMAPE(self.Xi(self.A, frames, alpha, beta), algo.Xi(self.A, frames, alpha, beta))
        self.assertLessEqual(smape, self.tolerance)

    def BaumWelch(self, O: npt.NDArray, A: npt.NDArray, mu: npt.NDArray, sigma: npt.NDArray, pi: npt.NDArray, tol: float, niter: int, min_sigma = 1e-5):
        (T, ) = O.shape
        (N, ) = mu.shape

        A = A.copy()
        mu = mu.copy()
        sigma = sigma.copy()
        pi = pi.copy()

        L = 1
        for _ in range(niter):
            frames = self.Frames(O, mu, sigma)
            alpha = self.Forward(A, frames, pi)
            beta = self.Backward(A, frames)
            delta = L - self.Likelihood(alpha)
            L = L + delta

            if np.absolute(delta) < tol:
                return (A, mu, sigma, pi)

            xi = self.Xi(A, frames, alpha, beta)
            gamma = self.Gamma(alpha, beta)

            for i in range(N):
                for j in range(N):
                    numerator = 0
                    for t in range(T-1):
                        numerator = numerator + xi[t, i, j]
                    
                    denominator = 0
                    for t in range(T-1):
                        denominator = denominator + gamma[t, i]
                    
                    A[i, j] = numerator / denominator

            for i in range(N):
                numerator = 0
                denominator = 0
                for t in range(T):
                    numerator = numerator + gamma[t, i] * np.power(O[t] - mu[i], 2)
                    denominator = denominator + gamma[t, i]

                sigma[i] = numerator / denominator
                if sigma[i] < min_sigma:
                    sigma[i] = min_sigma

            for i in range(N):
                numerator = 0
                denominator = 0
                for t in range(T):
                    numerator = numerator + gamma[t, i] * O[t]
                    denominator = denominator + gamma[t, i]
                
                mu[i] = numerator / denominator

            for i in range(N):
                pi[i] = gamma[0, i]

        return (A, mu, sigma, pi)


    def test_BaumWelch(self):
        tol = 1e-3
        niter = 100

        res = self.BaumWelch(self.O, self.A, self.mu, self.sigma, self.pi, tol, niter)
        algores = algo.BaumWelch(self.O, self.A, self.mu, self.sigma, self.pi, tol, niter)

        # A
        smape = self.SMAPE(res[0], algores[0])
        self.assertLessEqual(smape, self.tolerance)

        # mu
        smape = self.SMAPE(res[1], algores[1])
        self.assertLessEqual(smape, self.tolerance)

        # sigma
        smape = self.SMAPE(res[2], algores[2])
        self.assertLessEqual(smape, self.tolerance)

        # pi
        smape = self.SMAPE(res[3], algores[3])
        self.assertLessEqual(smape, self.tolerance)
