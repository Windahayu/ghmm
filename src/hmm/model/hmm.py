import numpy as np
import numpy.typing as npt

class BaseModel:
    A: npt.NDArray
    V: npt.NDArray
    B: npt.NDArray
    startprobs: npt.NDArray

    def __init__(self, A: npt.NDArray, V: npt.NDArray, B: npt.NDArray, startprobs: npt.NDArray):
        self.A = A.copy()
        self.V = V.copy()
        self.B = B.copy()
        self.startprobs = startprobs.copy()

    def Frames(self, O: npt.NDArray, V: npt.NDArray, B: npt.NDArray):
        (T, ) = O.shape
        (N, K) = B.shape

        frames = np.empty((N, T))
        for t in range(T):
            for k in range(K):
                if O[t] == V[k]:
                    frames[:, t] = B[:, k]
        
        return frames

    def Forward(self, A: npt.NDArray, frames: npt.NDArray, startprobs: npt.NDArray):
        (N, T) = frames.shape

        alpha = np.empty((T, N))
        for i in range(N):
            alpha[0, i] = startprobs[i] * frames[i, 0]
        for t in range(1, T):
            for j in range(N):
                x = 0 
                for i in range(N):
                    x = x + alpha[t-1, i] * A[i, j]
                alpha[t, j] = x * frames[j, t]

        return alpha

    def Backward(self, A: npt.NDArray, frames: npt.NDArray):
        (N, T) = frames.shape

        beta = np.empty((T, N))
        for i in range(N):
            beta[T - 1, i] = 1

        for t in range(T - 2, -1, -1):
            for i in range(N):
                x = 0
                for j in range(N):
                    x = x + A[i, j] * frames[j, t+1] * beta[t+1, j]
                beta[t, i] = x

        return beta

    def Likelihood(self, alpha: npt.NDArray):
        (T, N) = alpha.shape
        
        L = 0
        for i in range(N):
            L = L + alpha[T - 1, i]

        return L

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


    def Xi(self, A: npt.NDArray, frames: npt.NDArray, alpha: npt.NDArray, beta: npt.NDArray):
        (N, T) = frames.shape

        xi = np.empty((T-1, N, N))
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    numerator = alpha[t, i] * A[i, j] * frames[j, t+1] * beta[t+1, j]
                    denominator = 0
                    for k in range(N):
                        for l in range(N):
                            denominator = denominator + alpha[t, k] * A[k, l] * frames[l, t+1] * beta[t+1, l]
                    xi[t, i, j] = numerator / denominator
        
        return xi

    def RestimateA(self, xi: npt.NDArray, gamma: npt.NDArray):
        (T, N) = gamma.shape

        for i in range(N):
            for j in range(N):
                nominator = 0
                denominator = 0
                for t in range(T-1):
                    nominator = nominator + xi[t, i, j]
                    denominator = denominator + gamma[t, i]
                self.A[i,j] = nominator / denominator

    def RestimateB(self, O: npt.NDArray, gamma: npt.NDArray):
        (K, ) = self.V.shape
        (T, N) = gamma.shape

        for k in range(K):
            for i in range(N):
                numerator = 0
                denominator = 0
                for t in range(T):
                    if O[t] == self.V[k]:
                        numerator = numerator + gamma[t, i]
                    denominator = denominator + gamma[t, i]
                self.B[i, k] = numerator / denominator

    def RestimateStartProbs(self, gamma: npt.NDArray):
        (_, N) = gamma.shape

        for i in range(N):
            self.startprobs[i] = gamma[0, i]

    def Fit(self, O: npt.NDArray, tol: np.float_, niter: np.int_):
        L = 1
        for _ in range(niter):
            frames = self.Frames(O, self.V, self.B)
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

class Model(BaseModel):
    A: npt.NDArray
    V: npt.NDArray
    B: npt.NDArray
    startprobs: npt.NDArray

    def Frames(self, O: npt.NDArray, V: npt.NDArray, B: npt.NDArray):
        weight = (np.subtract.outer(O, V) == 0).astype(np.int8)
        frames = np.tensordot(B, weight, (1, 1))
        return frames

    def Forward(self, A: npt.NDArray, frames: npt.NDArray, startprobs: npt.NDArray):
        (N, T) = frames.shape

        alpha = np.empty((T, N))
        alpha[0] = startprobs * frames[:, 0]
        for t in range(1, T):
            alpha[t] = np.matmul(
                alpha[t-1], 
                A * frames[:, t]
            )

        return alpha

    def Backward(self, A: npt.NDArray, frames: npt.NDArray):
        (N, T) = frames.shape

        beta = np.empty((T, N))
        beta[-1] = np.ones((N ,))
        for t in range(T-2, -1, -1):
            beta[t] = np.matmul(
                frames[:, t+1] * beta[t+1],
                A.T,
            )

        return beta

    def Xi(self, A: npt.NDArray, frames: npt.NDArray, alpha: npt.NDArray, beta: npt.NDArray):
        (N, T) = frames.shape

        xi = np.empty((T-1, N, N))
        for t in range(T-1):
            xi[t] = ((A.T * alpha[t]).T * frames[:, t+1] * beta[t+1])
            xi[t] = xi[t] / xi[t].sum((0, 1))

        return xi

    def Gamma(self, alpha: npt.NDArray, beta: npt.NDArray):
        gamma = ((alpha * beta).T / (alpha * beta).sum(1)).T

        return gamma

    def Likelihood(self, alpha: npt.NDArray):
        L = alpha[-1].sum()

        return L

    def RestimateA(self, xi: npt.NDArray, gamma: npt.NDArray):
        self.A = (xi.sum(0).T / gamma[:-1].sum(0)).T

    def RestimateB(self, O: npt.NDArray, gamma: npt.NDArray):
        weight = (np.subtract.outer(O, self.V) == 0).astype(np.int8)
        self.B = (np.matmul(weight.T, gamma) / gamma.sum(0)).T
    
    def RestimateStartProbs(self, gamma: npt.NDArray):
        self.startprobs = gamma[0]
