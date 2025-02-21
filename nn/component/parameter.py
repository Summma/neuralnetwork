import numpy as np
import numpy.typing as npt

from nn.component.component import ParameterComponent


class Linear(ParameterComponent):
    def __init__(self, _in: int, _out: int) -> None:
        self.weights: npt.NDArray[np.float64] = np.random.randn(_out, _in) * np.sqrt(2 / _in)
        self.biases: npt.NDArray[np.float64] = np.random.rand(_out) / 10

        self.old_d_weights = np.zeros(self.weights.shape)
        self.old_d_biases = np.zeros(self.biases.shape)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.dot(self.weights, x) + self.biases

    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.weights.T

    def weight_derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.atleast_2d(x)

    def update_parameters(self, d_W: npt.NDArray[np.float64], d_B: npt.NDArray[np.float64], lr: float) -> None:
        # momentum_split = 0.3

        # self.weights -= lr * (d_W * (1 - momentum_split) + self.old_d_weights * momentum_split)
        # self.biases -= lr * (d_B.T[0] * (1 - momentum_split) + self.old_d_biases * momentum_split)

        # self.old_d_weights = momentum_split * self.old_d_weights + d_W
        # self.old_d_biases = momentum_split * self.old_d_biases + d_B.T[0]
        self.weights -= lr * d_W
        self.biases -= lr * d_B.T[0]
