import numpy as np
import numpy.typing as npt
from scipy.stats import norm

def Frames(O: npt.NDArray, means: npt.NDArray, variances: npt.NDArray):
    (T, ) = O.shape
    (N, ) = means.shape

    frames = np.empty((T, N))
    for i in range(T):
        frames[i] = norm.pdf(O[i], means, variances)
    
    return frames

def Forward(A: npt.NDArray, frames: npt.NDArray, pi: npt.NDArray):
    (T, N) = frames.shape

    alpha = np.empty((T, N))
    alpha[0] = pi * frames[0]
    for t in range(1, T):
        alpha[t] = np.matmul(
            alpha[t-1], 
            A * frames[t]
        )

    return alpha

def Backward(A: npt.NDArray, frames: npt.NDArray):
    (T, N) = frames.shape

    beta = np.empty((T, N))

    beta[-1] = np.ones((N ,))
    for t in range(T-2, -1, -1):
        beta[t] = np.matmul(
            frames[t+1] * beta[t+1],
            A.T,
        )

    return beta

def Likelihood(alpha: npt.NDArray):
    L = alpha[-1].sum()
    return L

def Xi(A: npt.NDArray, frames: npt.NDArray, alpha: npt.NDArray, beta: npt.NDArray):
    (T, N) = frames.shape

    xi = np.empty((T-1, N, N))
    for t in range(T-1):
        xi[t] = ((A.T * alpha[t]).T * frames[t+1] * beta[t+1])
        xi[t] = xi[t] / xi[t].sum((0, 1))

    return xi

def Gamma(alpha: npt.NDArray, beta: npt.NDArray):
    gamma = ((alpha * beta).T / (alpha * beta).sum(1)).T

    return gamma

def BaumWelch(O: npt.NDArray, A: npt.NDArray, means: npt.NDArray, variances: npt.NDArray, pi: npt.NDArray, tol: float, niter: int, min_variance = 1e-5):
    (T, ) = O.shape
    (N, ) = means.shape

    L = 1
    for _ in range(niter):
        frames = Frames(O, means, variances)
        alpha = Forward(A, frames, pi)
        beta = Backward(A, frames)
        delta = L - Likelihood(alpha)
        L = L + delta

        if np.absolute(delta) < tol:
            return (A, means, variances, pi)

        xi = Xi(A, frames, alpha, beta)
        gamma = Gamma(alpha, beta)

        A = (xi.sum(0).T / gamma[:T-1].sum(0)).T
        variances = (gamma * np.power(np.subtract.outer(O, means), 2)).sum(0) / gamma.sum(0)
        variances = np.maximum(variances, min_variance)
        means = (gamma.T * O).sum(1) / gamma.sum(0)
        pi = gamma[0]

    return (A, means, variances, pi)
