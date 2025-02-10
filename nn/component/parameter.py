import numpy as np

from nn.component.component import ParameterComponent


class Linear(ParameterComponent):
    def __init__(self, _in: int, _out: int) -> None:
        self.weights: np.ndarray = np.random.randn(_out, _in) * np.sqrt(2 / _in)
        self.biases: np.ndarray = np.random.rand(_out) / 10

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.weights, x) + self.biases

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.weights.T

    def weight_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.atleast_2d(x)

    def update_parameters(self, d_W: np.ndarray, d_B: np.ndarray, lr=0.001):
        self.weights -= lr * d_W
        self.biases -= lr * d_B.T[0]
