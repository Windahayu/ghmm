import numpy as np
import numpy.typing as npt
from scipy.stats import norm

class Model:
    A: npt.NDArray
    means: npt.NDArray
    variances: npt.NDArray
    startprob: npt.NDArray

    def __init__(self, A: npt.NDArray, means: npt.NDArray, variances: npt.NDArray, startprob: npt.NDArray):
        self.A = A
        self.means = means
        self.variances = variances
        self.startprob = startprob

    def Frames(O: npt.NDArray, means: npt.NDArray, variances: npt.NDArray):
        (T, ) = O.shape
        (N, ) = means.shape

        frames = np.empty((T, N))
        for i in range(T):
            frames[i] = norm.pdf(O[i], means, variances)
        
        return frames

    def Forward(A: npt.NDArray, B: npt.NDArray, startprob: npt.NDArray):
        (T, N) = B.shape

        alpha = np.empty((T, N))
        alpha[0] = startprob * B[0]
        for t in range(1, T):
            alpha[t] = np.matmul(
                alpha[t-1], 
                A * B[t]
            )

        return alpha

    def Backward(A: npt.NDArray, B: npt.NDArray):
        (T, N) = B.shape

        beta = np.empty((T, N))

        beta[-1] = np.ones((N ,))
        for t in range(T-2, -1, -1):
            beta[t] = np.matmul(
                B[t+1] * beta[t+1],
                A.T,
            )

        return beta

    def Xi(A: npt.NDArray, B: npt.NDArray, alpha: npt.NDArray, beta: npt.NDArray):
        (T, N) = B.shape

        xi = np.empty((T-1, N, N))
        for t in range(T-1):
            xi[t] = ((A.T * alpha[t]).T * B[t+1] * beta[t+1])
            xi[t] = xi[t] / xi[t].sum((0, 1))

        return xi

    def Gamma(alpha: npt.NDArray, beta: npt.NDArray):
        gamma = ((alpha * beta).T / (alpha * beta).sum(1)).T

        return gamma

    def Likelihood(alpha: npt.NDArray):
        L = alpha[-1].sum()

        return L

    def EstimateA(xi: npt.NDArray, gamma: npt.NDArray):
        A = (xi.sum(0).T / gamma[:-1].sum(0)).T

        return A

    def EstimateVariances(O: npt.NDArray, means: npt.NDArray, gamma: npt.NDArray, min_variance: np.float_):
        variances = (gamma * np.power(np.subtract.outer(O, means), 2)).sum(0) / gamma.sum(0)
        variances = np.maximum(variances, min_variance)

        return variances

    def EstimateMeans(O: npt.NDArray, gamma: npt.NDArray):
        means = (gamma.T * O).sum(1) / gamma.sum(0)

        return means
    
    def EstimateStratProbability(gamma: npt.NDArray):
        startprob = gamma[0]

        return startprob

    def Fit(self, O: npt.NDArray, min_variance: np.float_ = 1e-5, niter: np.int_ = 1000, tol: np.float_ = 1e-5):
        (T, ) = O.shape
        (N, ) = self.means.shape

        L = 1
        for _ in range(niter):
            B = Model.Frames(O, self.means, self.variances)
            alpha = Model.Forward(self.A, B, self.startprob)
            beta = Model.Backward(self.A, B)
            delta = L - Model.Likelihood(alpha)
            L = L + delta

            if np.absolute(delta) < tol:
                return

            xi = Model.Xi(self.A, B, alpha, beta)
            gamma = Model.Gamma(alpha, beta)

            self.startprob = Model.EstimateStratProbability(gamma)
            self.A = Model.EstimateA(xi, gamma)
            self.variances = Model.EstimateVariances(O, self.means, gamma, min_variance)
            self.means = Model.EstimateMeans(O, gamma)
