import numpy as np
import numpy.typing as npt
import pytest
from utils import SMAPE
from hmm.model.hmm import Model, BaseModel

class Args(object):
    T: int
    N: int
    K: int

    O: npt.NDArray
    A: npt.NDArray
    V: npt.NDArray
    B: npt.NDArray
    startprobs: npt.NDArray
    
    basemodel: BaseModel
    model: Model

    frames: npt.NDArray
    alpha: npt.NDArray
    beta: npt.NDArray
    xi: npt.NDArray
    gamma: npt.NDArray

    def __init__(self, O: npt.NDArray, A: npt.NDArray, V: npt.NDArray, B: npt.NDArray, startprobs: npt.NDArray, tol: np.float_, niter: np.int_):
        self.O = O.copy()
        self.A = A.copy()
        self.V = V.copy()
        self.B = B.copy()
        self.startprobs = startprobs.copy()
        self.tol = tol
        self.niter = niter

        self.pre_calculate()

    def pre_calculate(self):
        (self.T, ) = self.O.shape
        (self.N, self.K) = self.B.shape

        args = (self.A, self.V, self.B, self.startprobs)
        self.basemodel = BaseModel(*args)
        self.model = Model(*args)

        self.frames = self.basemodel.Frames(self.O, self.V, self.B)
        self.alpha = self.basemodel.Forward(self.A, self.frames, self.startprobs)
        self.beta = self.basemodel.Backward(self.A, self.frames)

        self.xi = self.basemodel.Xi(self.A, self.frames, self.alpha, self.beta)
        self.gamma = self.basemodel.Gamma(self.alpha, self.beta)

@pytest.fixture
def args():
    T = 4
    N = 3
    K = 5

    O = np.array(range(T), dtype=np.float_)
    A = np.ones((N, N), dtype=np.float_) / N
    V = np.array(range(K))
    B = np.ones((N, K), dtype=np.float_) / N
    startprobs = np.ones((N), dtype=np.float_) / N

    tol = np.float_(1e-5)
    niter = np.int_(100)

    args = Args(O, A, V, B, startprobs, tol, niter)
    return args

tolerance = 1e-10

def test_Frames(args):
    test_args = (args.O, args.V, args.B)
    smape = SMAPE(args.basemodel.Frames(*test_args), args.model.Frames(*test_args))
    assert smape <= tolerance

def test_Forward(args):
    test_args = (args.A, args.frames, args.startprobs)
    smape = SMAPE(args.basemodel.Forward(*test_args), args.model.Forward(*test_args))
    assert smape <= tolerance

def test_Backward(args):
    test_args = (args.A, args.frames)
    smape = SMAPE(args.basemodel.Backward(*test_args), args.model.Backward(*test_args))
    assert smape <= tolerance

def test_Likelihood(args):
    test_args = (args.alpha, )
    smape = SMAPE(np.array([args.basemodel.Likelihood(*test_args)]), np.array([args.model.Likelihood(*test_args)]))
    assert smape <= tolerance

def test_Gamma(args):
    test_args = (args.alpha, args.beta)
    smape = SMAPE(args.basemodel.Gamma(*test_args), args.model.Gamma(*test_args))
    assert smape <= tolerance

def test_Xi(args):
    test_args = (args.A, args.frames, args.alpha, args.beta)
    smape = SMAPE(args.basemodel.Xi(*test_args), args.model.Xi(*test_args))
    assert smape <= tolerance

def test_RestimateA(args):
    test_args = (args.xi, args.gamma)
    args.model.RestimateA(*test_args)
    args.basemodel.RestimateA(*test_args)
    smape = SMAPE(args.basemodel.A, args.model.A)
    assert smape <= tolerance

def test_RestimateB(args):
    test_args = (args.O, args.gamma)
    args.model.RestimateB(*test_args)
    args.basemodel.RestimateB(*test_args)
    smape = SMAPE(args.basemodel.B, args.model.B)
    assert smape <= tolerance

def test_RestimateStartProbs(args):
    test_args = (args.gamma, )
    args.model.RestimateStartProbs(*test_args)
    args.basemodel.RestimateStartProbs(*test_args)
    smape = SMAPE(args.basemodel.startprobs, args.model.startprobs)
    assert smape <= tolerance

def test_Fit(args):
    test_args = (args.O, args.tol, args.niter)
    args.model.Fit(*test_args)
    args.basemodel.Fit(*test_args)
    smape_A = SMAPE(args.basemodel.A, args.model.A)
    assert smape_A <= tolerance
    smape_B = SMAPE(args.basemodel.B, args.model.B)
    assert smape_B <= tolerance
    smape_startprobs = SMAPE(args.basemodel.startprobs, args.model.startprobs)
    assert smape_startprobs <= tolerance

pytestmark = pytest.mark.benchmark(group = "Hidden Markov Model")

def test_BaseModel(args, benchmark):
    benchmark(args.basemodel.Fit, args.O, args.tol, args.niter)

def test_Model(args, benchmark):
    benchmark(args.model.Fit, args.O, args.tol, args.niter)
