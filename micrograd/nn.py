import random
from micrograd.engine import Value

class Module:

    def zero_grad(self): # Needed to reset gradients after run
        for p in self.parameters():
            p.grad = 0

    def parameters(self): # Placeholder, just here so the code wont crash if it's not defined later
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True): # nin = Number of inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)  # zip combines w and x together into (wi, xi) for each pair
                                                                # Then returns the sum of all of them (typical MLP)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b] # Returns a list of variables (which then get reset)

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs): # nin = Number of inputs, nout = Number of outputs
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)] # Each Neuron in nout has nin input layers, so the neurons here
                                                                    # are the amount of output neurons (makes sense)
    # Example with Layer(10, 5), this layer has 5 neurons that each have an input of 10 neurons

    def __call__(self, x): # A call happens, if the object is already defined, for example: n = Neuron(2); out = n([2, 3])
        out = [n(x) for n in self.neurons] # So this works because __call__ isn't __init__ (makes sense...)
        return out[0] if len(out) == 1 else out # Quality of Life, if it's the last layer and the output has one value, it won't be in a list this way

    def parameters(self): # Lists all values (parameters) to reset gradient
        return [p for n in self.neurons for p in n.parameters()] # The self.neurons loop happens first, then the parameter one... (the p at the start is left for last though)

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts): # nouts is a list of outputs
        sz = [nin] + nouts # combines a list with a list of lists to get a list of lists...
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]    # List of layers, recursively through all layers. The nonlin makes sure, that the last layer doesn't use ReLu
                                                                                                    # because otherwise the last neuron never could be negative (makes sense)
    def __call__(self, x): # Calling MLP with input x
        for layer in self.layers: # Uses input x and goes through each layer
            x = layer(x)
        return x # Now x from output layer

    def parameters(self): # Same structure as with Layer
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
