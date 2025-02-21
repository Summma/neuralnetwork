import numpy as np
import numpy.typing as npt


class MSE:
    @staticmethod
    def calculate(y: npt.NDArray[np.float64], y_hat: npt.NDArray[np.float64]):
        return np.mean((y - y_hat) ** 2)

    @staticmethod
    def derivative(y: npt.NDArray[np.float64], y_hat: npt.NDArray[np.float64]):
        return -2 * np.mean(y - y_hat)
