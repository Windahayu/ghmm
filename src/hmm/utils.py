import numpy as np
import numpy.typing as npt

def MAPE(M1: npt.NDArray, M2: npt.NDArray):
    mape = (np.absolute(M1 - M2) / M1).mean() * 1e+2
    return mape
