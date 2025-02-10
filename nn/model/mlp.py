import numpy as np

from nn.component.component import Component, ParameterComponent, ActivationComponent
from nn.loss import MSE


class MLP:
    def __init__(self, *args: Component) -> None:
        self.components: tuple[Component, ...] = args
        self.loss = MSE()

    def forward_pass(self, x: np.ndarray):
        y = x
        for component in self.components:
            y = component.forward(y)

        return y

    def _forward_pass_with_outputs(self, x: np.ndarray):
        y = x
        outputs = [x]
        for component in self.components:
            y = component.forward(y)
            outputs.append(y)

        return outputs
    
    def cost(self, x: np.ndarray, y: np.ndarray) -> float:
        y_hat = self.forward_pass(x)
        return self.loss.calculate(y, y_hat)

    
    def back_prop(self, x: np.ndarray, y: np.ndarray):
        outputs = self._forward_pass_with_outputs(x)
        y_hat = outputs.pop()

        reuse = self.loss.derivative(y, y_hat) * np.identity(y_hat.shape[0])

        for i, component in list(enumerate(self.components))[-2::-2]:
            activation = self.components[i + 1]
            layer = self.components[i]

            activation_input = outputs[i + 1]
            layer_input = outputs[i]

            if isinstance(layer, ParameterComponent) and isinstance(activation, ActivationComponent):
                a_der = np.atleast_2d(activation.derivative(activation_input)).T
                w_der = layer.weight_derivative(layer_input)

                precomp = np.sum(reuse, axis=1)
                precomp = np.atleast_2d(precomp).T

                weight_updates = a_der @ w_der * precomp
                bias_updates = a_der * precomp
            else:
                raise Exception("Network Structure Likely Incorrect. Should Follow:\n\t Parameters > Activation > Parameters > Activation > ... > Activation")


            reuse = layer.derivative(layer_input) @ activation.derivative_matrix(activation_input) @ reuse
            layer.update_parameters(weight_updates, bias_updates, lr=0.0010115)
