import numpy as np
import numpy.typing as npt
from scipy.stats import norm

def Frames(O: npt.NDArray, mu: npt.NDArray, sigma: npt.NDArray):
    (T, ) = O.shape
    (N, ) = mu.shape

    frames = np.empty((T, N))
    for i in range(T):
        frames[i] = norm.pdf(O[i], mu, sigma)
    
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

    L = Likelihood(alpha)

    xi = np.empty((T-1, N, N))
    for t in range(T-1):
        xi[t] = ((alpha[t] * A.T).T * frames[t+1] * beta[t+1]) / L

    return xi

def Gamma(alpha: npt.NDArray, beta: npt.NDArray):
    gamma = ((alpha * beta).T / (alpha * beta).sum(1)).T

    return gamma

def BaumWelch(O: npt.NDArray, A: npt.NDArray, mu: npt.NDArray, sigma: npt.NDArray, pi: npt.NDArray, tol: float, niter: int):
    (T, ) = O.shape
    (N, ) = mu.shape

    L = 0
    for _ in range(niter):
        frames = Frames(O, mu, sigma)
        alpha = Forward(A, frames, pi)
        beta = Backward(A, frames)
        delta = L - Likelihood(alpha)
        L = L + delta

        if delta < tol:
            return (A, mu, sigma, pi)

        xi = Xi(A, frames, alpha, beta)
        gamma = Gamma(alpha, beta)

        A = xi.sum(0) / np.tile(gamma[:T - 1].sum(0), (N, 1)).T
        sigma = (gamma * np.power((np.tile(O, (N, 1)).T - mu), 2)).sum(0) / gamma.sum(0)
        mu = (gamma.T * O).sum(1) / gamma.sum(0)
        pi = gamma[1]

    return (A, mu, sigma, pi)
