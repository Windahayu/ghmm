import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from hmm.model import hmm

class BaseModel(hmm.BaseModel): 
    A: npt.NDArray
    means: npt.NDArray
    variances: npt.NDArray
    startprob: npt.NDArray
    min_variance: np.float_

    def __init__(self, A: npt.NDArray, means: npt.NDArray, variances: npt.NDArray, startprob: npt.NDArray, min_variance: np.float_ = np.float_(1e-2)):
        self.A = A.copy()
        self.means = means.copy()
        self.variances = variances.copy()
        self.startprob = startprob.copy()
        self.min_variance = min_variance.copy()

    def Frames(self, O: npt.NDArray, means: npt.NDArray, variances: npt.NDArray):
        (T, ) = O.shape
        (N, ) = means.shape

        frames = np.empty((N, T))
        for i in range(N):
            for t in range(T):
                frames[i, t] = norm.pdf(O[t], means[i], variances[i])

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

class Model(hmm.Model):
    A: npt.NDArray
    means: npt.NDArray
    variances: npt.NDArray
    startprob: npt.NDArray
    min_variance: np.float_

    def __init__(self, A: npt.NDArray, means: npt.NDArray, variances: npt.NDArray, startprob: npt.NDArray, min_variance: np.float_ = np.float_(1e-2)):
        self.A = A.copy()
        self.means = means.copy()
        self.variances = variances.copy()
        self.startprob = startprob.copy()
        self.min_variance = min_variance.copy()

    def Frames(self, O: npt.NDArray, means: npt.NDArray, variances: npt.NDArray):
        (T, ) = O.shape
        (N, ) = means.shape

        frames = np.empty((N, T))
        for t in range(T):
            frames[:, t] = norm.pdf(O[t], means, variances)
        
        return frames

    def RestimateB(self, O: npt.NDArray, gamma: npt.NDArray):
        self.variances = (gamma * np.power(np.subtract.outer(O, self.means), 2)).sum(0) / gamma.sum(0)
        self.variances = np.maximum(self.variances, self.min_variance)

        self.means = (gamma.T * O).sum(1) / gamma.sum(0)
