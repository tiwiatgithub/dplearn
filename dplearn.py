import numpy as np

class Model(object):

    def __init__(self):

        self.nb_layers    = 0
        self.nb_inputs    = 0
        self.input_values = 0
        self.layers       = list()
        self.size_layers  = list()
        self.weights      = list()
        self.biases       = list()
        self.outputs      = list()
        weighted_inputs   = list()
        self.gradients    = list()
        self.activations  = list()
        self.loss_function= []
        self.iter         = 0

    def add(self, layer):
        if layer.layer_type == 'dense':
            self.layers.append(layer)
            self.size_layers.append(layer.nb_neurons)
            self.activations.append(layer.activation)
            self.nb_layers += 1
            self.X = np.ndarray
        elif layer.layer_type == 'input':
            self.nb_inputs = layer.nb_neurons

    def setup(self, loss='sgd'):
        size_total = [self.nb_inputs] + self.size_layers # size of all layers including input layer
        print("size total: %s" % size_total)
        self.weighted_inputs   = [np.zeros((x, 1)) for x in self.size_layers]
        self.outputs           = [np.zeros((x, 1)) for x in self.size_layers]
        self.biases            = [np.ones((x, 1))  for x in self.size_layers]
        self.weights           = [np.random.randn(x,y) for x,y in zip(size_total[:-1], size_total[1:])]
        self.gradients         = [np.zeros((x, y))     for x,y in zip(size_total[:-1], size_total[1:])]
        self.loss_function     = []
        #print("w0: %d" %  self.weights[0])
        #print("w1: %d" %  self.weights[1])



    def print_model_parameters(self):
        print("network size: \n%s" % (self.size_layers))
        for weight in self.weights:
            print("weights: \n%s" % (weight))
        for bias in self.biases:
            print("biases: \n%s" % (bias))
        for weighted_input in self.weighted_inputs:
            print("z: \n%s" % (weighted_input))

        for output in self.outputs:
            print("outputs: \n%s" % (output))
        for gradient in self.gradients:
            print("gradients: \n%s" % (gradient))
        for activation in self.activations:
            print("activations: \n%s" % activation)


    def forward(self, x, verbose=0):
        print("propagate inputs forward ...")
        self.input_values = x
        for l in range(self.nb_layers):
            if verbose == 1: print("inside model.forward.")
            w = self.weights[l]
            b = self.biases[l]
            z     = np.dot(w.T, x) + b
            sigma = self.activations[l].forward(z)
            if verbose == 1:
                print("x[l-1]: %s" % [x.shape])
                print("w: %s" % [w.shape])
                print("b: %s" % [b.shape])
                print("z: %s" % [z.shape])
                print("sigma: %s" % [sigma.shape])
            x = sigma
            if verbose == 1: print("x[l]: %s" % [x.shape])

            self.weighted_inputs[l] = z
            self.outputs[l] = sigma

    def backward(self, grad, verbose=0):
        print("propagate gradients backward ...")
        weighted_inputs = self.weighted_inputs
        biases = self.biases
        outputs = self.outputs
        activations = self.activations
        gradients = self.gradients
        x = self.input_values

        for l in reversed(range(self.nb_layers)):
            if verbose == 1:print("l: %i" % l)
            if l > 0:
                sigma = outputs[l-1]
                a = activations[l].backward(weighted_inputs[l])
                d = np.sum(grad)

                gradients[l] = sigma * a.T * d
                grad = gradients[l]
            elif l == 0:
                a = activations[l].backward(weighted_inputs[l])
                d = np.sum(grad)
                gradients[l] = x * a.T * d

                grad = gradients[l]
        self.gradients = gradients

    def update(self, lr=0.01):
        print("update weights ...")

        for l in range(self.nb_layers):
            if l > 0:
                self.weights[l] -=  lr * self.outputs[l-1] *  self.gradients[l]
                self.biases[l]  -=  lr * self.gradients[l]
            elif l == 0:
                self.weights[l] -=  lr * self.input_values *  self.gradients[l]
                self.biases[l]  -=  lr * self.gradients[l]

    def train(self, x, y, iter_max=10):
        self.iter = 0
        dummy_grad = 0
        while self.iter < iter_max:
            self.forward(x)
            y_hat = self.outputs[-1]
            self.loss(y, y_hat)
            self.backward(dummy_grad)
            self.update()
            self.iter +=1

    def loss(self, y, y_hat):
        return -2*(y - y_hat)

class Layer(object):
    def __init__(self, nb_neurons=1, layer_type='dense', activation='relu'):
        self.nb_neurons = nb_neurons
        self.activation = Activation.get(name=activation)
        self.layer_type = layer_type


class Activation(object):

    def get(name='relu'):
        if name == 'relu':    return Relu()
        if name == 'sigmoid': return Sigmoid()
    get = staticmethod(get)

class Relu(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x):
        print(x.shape)
        result = np.ones(x.shape)
        result[x<0] = 0

        print("inside relu.backward: %s" % result)
        return result

class Sigmoid(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return 1/ (1 + np.exp(-x))

    def backward(self, x):
        print("inside sigmoid.backward ...")
        result = (self.forward(x) - self.forward(x)**2)
        return result

class Optimizer(object):

    def get(name='sgd'):
        if name == 'sgd': return SGD()

class SGD(Optimizer):
    def __init__(self): pass

class Metric(object):
    def get(name='squared_error'):
        if name == 'squared_error': return 0
    get = staticmethod(get)

class SquaredError(Metric):
    def __init__(self):
        self.y_errs      = list()
        self.loss_values = list()
        self.gradients   = list()

    def loss(self, y_real, y_hat):
        self.y_err.append(y_real - y_hat)
        self.loss_values.append(np.dot(self.y_errs[-1].T, self.y_errs[-1]))
        return self.loss_values[-1]

    def gradient(self):
        self.gradients.append(-2 * self.errs[-1])
        return self.gradients[-1]
