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
