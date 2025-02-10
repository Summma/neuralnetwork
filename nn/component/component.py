from abc import ABC, abstractmethod

import numpy as np

class Component(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
        
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class ActivationComponent(Component):
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

    def derivative_matrix(self, x: np.ndarray) -> np.ndarray:
        derivative = self.derivative(x)
        return np.diag(derivative)


class ParameterComponent(Component):
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def weight_derivative(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_parameters(self, d_W: np.ndarray, d_B: np.ndarray):
        pass
