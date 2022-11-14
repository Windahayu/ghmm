import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from hmm.model import hmm

class BaseModel(hmm.BaseModel): 
    A: npt.NDArray
    means: npt.NDArray
    variances: npt.NDArray
    startprobs: npt.NDArray
    min_variance: np.float_

    def __init__(self, A: npt.NDArray, means: npt.NDArray, variances: npt.NDArray, startprob: npt.NDArray, min_variance: np.float_ = np.float_(1e-2)):
        self.A = A.copy()
        self.means = means.copy()
        self.variances = variances.copy()
        self.startprobs = startprob.copy()
        self.min_variance = min_variance.copy()

    def Frames(self, O: npt.NDArray, means: npt.NDArray, variances: npt.NDArray):
        (T, ) = O.shape
        (N, ) = means.shape

        frames = np.empty((N, T))
        for i in range(N):
            for t in range(T):
                frames[i, t] = norm.pdf(O[t], means[i], np.sqrt(variances[i]))

        return frames

    def RestimateB(self, O: npt.NDArray, gamma: npt.NDArray):
        (T, ) = O.shape
        (N, ) = self.means.shape

        for i in range(N):
            numerator = 0
            denominator = 0
            for t in range(T):
                numerator = numerator + gamma[t, i] * np.power(O[t] - self.means[i], 2)
                denominator = denominator + gamma[t, i]
            
            self.variances[i] = numerator / denominator
            if self.variances[i] < self.min_variance:
                self.variances[i] = self.min_variance

        for i in range(N):
            numerator = 0
            denominator = 0
            for t in range(T):
                numerator = numerator + gamma[t, i] * O[t]
                denominator = denominator + gamma[t, i]
            
            self.means[i] = numerator / denominator

    def Fit(self, O: npt.NDArray, tol: np.float_, niter: np.int_):
        L = 1
        for _ in range(niter):
            frames = self.Frames(O, self.means, self.variances)
            alpha = self.Forward(self.A, frames, self.startprobs)
            beta = self.Backward(self.A, frames)
            delta = L - self.Likelihood(alpha)
            L = L + delta

            if np.absolute(delta) < tol:
                return

            xi = self.Xi(self.A, frames, alpha, beta)
            gamma = self.Gamma(alpha, beta)

            self.RestimateA(xi, gamma)
            self.RestimateB(O, gamma)
            self.RestimateStartProbs(gamma)

class Model(hmm.Model):
    A: npt.NDArray
    means: npt.NDArray
    variances: npt.NDArray
    startprobs: npt.NDArray
    min_variance: np.float_

    def __init__(self, A: npt.NDArray, means: npt.NDArray, variances: npt.NDArray, startprob: npt.NDArray, min_variance: np.float_ = np.float_(1e-2)):
        self.A = A.copy()
        self.means = means.copy()
        self.variances = variances.copy()
        self.startprobs = startprob.copy()
        self.min_variance = min_variance.copy()

    def Frames(self, O: npt.NDArray, means: npt.NDArray, variances: npt.NDArray):
        frames = np.exp((np.power(np.subtract.outer(O, means), 2) / variances + np.log(2 * np.pi) + np.log(variances)).T / -2)
        return frames

    def RestimateB(self, O: npt.NDArray, gamma: npt.NDArray):
        self.variances = (gamma * np.power(np.subtract.outer(O, self.means), 2)).sum(0) / gamma.sum(0)
        self.variances = np.maximum(self.variances, self.min_variance)

        self.means = (gamma.T * O).sum(1) / gamma.sum(0)

    def Fit(self, O: npt.NDArray, tol: np.float_, niter: np.int_):
        L = 1
        for _ in range(niter):
            frames = self.Frames(O, self.means, self.variances)
            alpha = self.Forward(self.A, frames, self.startprobs)
            beta = self.Backward(self.A, frames)
            delta = L - self.Likelihood(alpha)
            L = L + delta

            if np.absolute(delta) < tol:
                return

            xi = self.Xi(self.A, frames, alpha, beta)
            gamma = self.Gamma(alpha, beta)

            self.RestimateA(xi, gamma)
            self.RestimateB(O, gamma)
            self.RestimateStartProbs(gamma)

