from typing import Callable

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from nn.component.parameter import Linear
from nn.component.activation import ReLU, Identity
from nn.model.mlp import MLP

np.random.seed(50)


if __name__ == "__main__":
    # test()
    # exit(1)
    # model = MLP(Linear(2, 3), ReLU(), Linear(3, 3), ReLU(), Linear(3, 50), ReLU(), Linear(50, 3), ReLU())
    model = MLP(Linear(1, 50), ReLU(), Linear(50, 300), ReLU(), Linear(300, 1), Identity())

    f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = lambda z: 3 * np.sin(3 * z)

    # X = 4 * np.random.rand(100, 1) - 2
    X = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y = f(X)

    losses: list[np.floating] = []

    for i in range(5000):
        for x, _y in zip(X, y):
            loss = model.cost(x, _y)
            model.back_prop(x, _y, lr=0.001)

            losses.append(loss)

        if (i + 1) % 100 == 0:
            print(f"Epoch: {i}, Loss: {sum(losses) / len(losses)}")

    plt.plot([sum(losses[i:i+1000]) / 1000 for i in range(0, len(losses), 1000)])
    plt.show()

    print(model.forward_pass(np.array([1.45])))
    print(losses[-1])
    print(sum(losses) / len(losses))
    m = model.components[0].weights[0][0]
    b = model.components[0].biases[0]

    # X = np.sort(X, axis=0)
    #
    # y = f(X)
    plt.plot(np.sort(X), np.sort(y))

    y = np.array([model.forward_pass(x) for x in X])
    plt.plot(X, y)

    plt.show()
