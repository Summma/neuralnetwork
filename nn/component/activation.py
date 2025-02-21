import numpy as np
import numpy.typing as npt

from nn.component.component import  ActivationComponent


class ReLU(ActivationComponent):
    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.maximum(x, 0)

    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(x > 0, 1, 0)


class Identity(ActivationComponent):
    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x

    def derivative(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.ones(x.shape[0])
