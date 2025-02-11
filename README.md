# Custom Neural Network

Custom neural made with very loose inspiration from PyTorch. <br />
## Default Usage
The main.py file shows an example of usage, but the basic idea is as follows:
```python
import numpy as np

from nn.component.parameter import Linear
from nn.component.activation import ReLU, Identity
from nn.model.mlp import MLP


# Initialize network
model = MLP(Linear(1, 3), ReLU(), Linear(3, 1), Identity()) # The reason for Identity() is because the network
                                                            # expects an activation for every Linear layer
f = lambda z: 3 * np.sin(20 * z)

X = np.random.rand(100000, 1)
y = f(X)

losses = []

for i in range(10):
    for x, _y in zip(X, y):
        # Calculates loss for single data point
        loss = model.cost(x, _y)

        # Model performs backpropagation and weights updates simultaneously
        model.back_prop(x, _y)
        
        losses.append(loss)
    print(f"Epoch: {i}, Loss: {losses[-1]}")
```

The following graph shows the network in main.py's approximation of f(X).
![alt text](https://github.com/Summma/neuralnetwork/blob/main/network_graph.jpg)

## Backpropagation Math
The general principle behind backpropagation lies in the chain rule. For example to derive $\frac{\partial C}{\partial W_0}$, you must do: <br />
```math
\frac{\partial C}{\partial W_0} = \frac{\partial z_0}{\partial W_0} \cdot \frac{\partial a_0}{\partial z_0}
\dots \frac{\partial a_k}{\partial z_{k-1}} \cdot \frac{\partial z_k}{\partial a_k} \cdot \frac{\partial C}{\partial z_k}
```
But this has some issues, as $z_0$ is a vector containing the first layer's output, and $W_0$ is a matrix. This means the corresponding
partial derivative returns a 3D tensor, which is entirely unnecessary. All of the information in that 3D tensor can be contained within
a matrix, which allows for faster calculation of the cost gradient. Let's assume we're working on the $k^{th}$ layer of our network.
We can turn $\frac{\partial z_k}{\partial W_k} \cdot \frac{\partial a_k}{\partial z_k}$ 
into $\frac{\partial a_0}{\partial z_0} \cdot x^T$, where $x$ is our input vector. <br /><br />
***This swapping of the order is the reason I decided to do the computations in pairs, otherwise it was too confusing.*** <br /><br />
However, this also changes the shape of the output from a 3D tensor to a matrix,
which means we have to change some of the rest of the equation as well. Each $\frac{\partial z_{k+1}}{\partial a_k} \cdot \frac{\partial a_{k+1}}{\partial z_{k+1}}$ pair in
the chain rule can simplify to $W_{k+1}^T \cdot \frac{\partial a_{k+1}}{\partial z_{k+1}}$, and we multiply these pairs going all the way up to final layer. The result will return
a matrix, $V_k$, with the same number of rows, $m$, as $W_k$, and a number of columns, $n$, equaling the dimension of the network's output layer. Due to $\frac{\partial a_0}{\partial z_0} \cdot x^T$
no longer being a 3D tensor, we have to add up all the columns in $V_k$--which is essentially the same as adding up how $W_k$ individually changes each value in the
output layer--and perform an element-wise multiplication $(\frac{\partial a_0}{\partial z_0} \cdot x^T) \odot (\sum V_k)$, which looks like:
```math
\begin{align}
  V_k = \prod_{i=k+1}^{\text{Last Layer}} W_{i}^T \cdot \frac{\partial a_{i}}{\partial z_{i}} \\
  \frac{\partial C}{\partial W_k} = (\frac{\partial a_0}{\partial z_0} \cdot x^T) \odot \sum_{i=1}^n V_k[:, i]
\end{align}
```
<br />

$V_k$ can and should also be updated incrementally as you work backward in the network, which allows it to perform backpropagation much more quickly.
In the code itself, this saved precomputation is called `reuse`.
