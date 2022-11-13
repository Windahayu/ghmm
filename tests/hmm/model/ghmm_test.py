import numpy as np
import numpy.typing as npt
import pytest
from utils import SMAPE
from hmm.model.ghmm import Model, BaseModel

class Args(object):
    T: int
    N: int

    O: npt.NDArray
    A: npt.NDArray
    means: npt.NDArray
    variances: npt.NDArray
    startprobs: npt.NDArray
    min_variance: np.float_

    basemodel: BaseModel
    model: Model

    frames: npt.NDArray
    alpha: npt.NDArray
    beta: npt.NDArray
    xi: npt.NDArray
    gamma: npt.NDArray

    def __init__(self, O: npt.NDArray, A: npt.NDArray, means: npt.NDArray, variances: npt.NDArray, startprobs: npt.NDArray, tol: np.float_, niter: np.int_, min_variance: np.float_):
        self.O = O.copy()
        self.A = A.copy()
        self.means = means.copy()
        self.variances = variances.copy()
        self.startprobs = startprobs.copy()
        self.tol = tol
        self.niter = niter
        self.min_variance = min_variance

        self.pre_calculate()

    def pre_calculate(self):
        (self.T, ) = self.O.shape
        (self.N, ) = self.means.shape

        args = (self.A, self.means, self.variances, self.startprobs)
        self.basemodel = BaseModel(*args)
        self.model = Model(*args)

        self.frames = self.basemodel.Frames(self.O, self.means, self.variances)
        self.alpha = self.basemodel.Forward(self.A, self.frames, self.startprobs)
        self.beta = self.basemodel.Backward(self.A, self.frames)

        self.xi = self.basemodel.Xi(self.A, self.frames, self.alpha, self.beta)
        self.gamma = self.basemodel.Gamma(self.alpha, self.beta)

@pytest.fixture
def args():
    T = 4
    N = 3

    O = np.array(range(T), dtype=np.float_)
    A = np.ones((N, N), dtype=np.float_) / N
    means = np.zeros((N, ), dtype=np.float_)
    variances = np.ones((N, ), dtype=np.float_) / N
    startprobs = np.ones((N), dtype=np.float_) / N

    tol = np.float_(1e-5)
    niter = np.int_(100)
    min_variance = np.float_(1e-2)

    args = Args(O, A, means, variances, startprobs, tol, niter, min_variance)
    return args

tolerance = 1e-10

def test_Frames(args):
    test_args = (args.O, args.means, args.variances)
    smape = SMAPE(args.basemodel.Frames(*test_args), args.model.Frames(*test_args))
    assert smape <= tolerance

def test_RestimateB(args):
    test_args = (args.O, args.gamma)
    args.model.RestimateB(*test_args)
    args.basemodel.RestimateB(*test_args)
    smape_means = SMAPE(args.basemodel.means, args.model.means)
    assert smape_means <= tolerance
    smape_variances = SMAPE(args.basemodel.variances, args.model.variances)
    assert smape_variances <= tolerance

pytestmark = pytest.mark.benchmark(group = "Gaussian Hidden Markov Model")

def test_BaseModel(args, benchmark):
    benchmark(args.basemodel.Fit, args.O, args.tol, args.niter)

def test_Model(args, benchmark):
    benchmark(args.model.Fit, args.O, args.tol, args.niter)
