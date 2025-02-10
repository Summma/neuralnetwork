import numpy as np

from nn.component.component import  ActivationComponent


class ReLU(ActivationComponent):
    def forward(self, x: np.ndarray):
        return np.maximum(x, 0)

    def derivative(self, x: np.ndarray):
        return np.where(x > 0, 1, 0)


class Identity(ActivationComponent):
    def forward(self, x: np.ndarray):
        return x
    
    def derivative(self, x: np.ndarray):
        return np.ones(x.shape[0])
