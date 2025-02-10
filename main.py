import numpy as np
import matplotlib.pyplot as plt

from nn.component.parameter import Linear
from nn.component.activation import ReLU, Identity
from nn.model.mlp import MLP

np.random.seed(1)



if __name__ == "__main__":
    # model = MLP(Linear(2, 3), ReLU(), Linear(3, 3), ReLU(), Linear(3, 50), ReLU(), Linear(50, 3), ReLU())
    model = MLP(Linear(1, 30), ReLU(), Linear(30, 10), ReLU(), Linear(10, 1), Identity())

    f = lambda z: 3 * np.sin(20 * z)
    
    X = np.random.rand(100000, 1)
    y = f(X)

    losses = []

    for i in range(10):
        for x, _y in zip(X, y):
            loss = model.cost(x, _y)
            model.back_prop(x, _y)
            
            losses.append(loss)
        print(f"Epoch: {i}, Loss: {losses[-1]}")

    plt.plot(losses)
    plt.show()

    print(model.forward_pass(np.array([1.45])))
    print(losses[-1])
    m = model.components[0].weights[0][0]
    b = model.components[0].biases[0]

    X = np.sort(X[:300], axis=0)

    y = f(X)
    plt.plot(np.sort(X), np.sort(y))

    y = np.array([model.forward_pass(x) for x in X])
    plt.plot(X, y)

    plt.show()
