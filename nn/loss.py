import numpy as np


class MSE:
    @staticmethod
    def calculate(y: np.ndarray, y_hat: np.ndarray):
        return np.mean((y - y_hat) ** 2)
    
    @staticmethod
    def derivative(y: np.ndarray, y_hat: np.ndarray):
        return -2 * np.mean(y - y_hat)
