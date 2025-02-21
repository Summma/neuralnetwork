from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class Component(ABC):
    @abstractmethod
    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass


class ActivationComponent(Component):
    @abstractmethod
    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    def derivative_matrix(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        derivative = self.derivative(x)
        return np.diag(derivative)


class ParameterComponent(Component):
    @abstractmethod
    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def weight_derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def update_parameters(self, d_W: npt.NDArray[np.float64], d_B: npt.NDArray[np.float64], lr: float):
        pass
