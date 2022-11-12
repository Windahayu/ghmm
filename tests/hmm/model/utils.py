import numpy as np
import numpy.typing as npt

def SMAPE(M1: npt.NDArray, M2: npt.NDArray):
    smape = (np.absolute(M1 - M2) / (np.absolute(M1) + np.absolute(M2)).mean()).mean() * 1e+2
    return smape
