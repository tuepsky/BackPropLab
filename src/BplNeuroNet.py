import scipy.special
import numpy as np


def log(*a):
    # print(*a)
    pass


class Layer(object):

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.w = None
        self.b = None
        self.previous = None
        self.excitation = None
        self.error = None

    def set(self, _pattern):
        self.excitation = _pattern

    def get_excitation(self):
        return self.excitation

    def get_error(self):
        ret_val = np.dot(self.w.transpose(), self.error)
        return ret_val

    def forward(self):
        _sum = np.dot(self.w, self.previous.get_excitation()) + self.b
        self.excitation = scipy.special.expit(_sum)

    def connect(self, previous):
        self.previous = previous
        self.w = np.random.normal(0.0, pow(previous.size, -0.5), (self.size, previous.size))
        self.b = np.random.normal(0.0, pow(previous.size, -0.5), self.size)

    def backward(self, err, alpha):
        self.error = err
        # Calculate adjustment factors
        val_tmp = self.excitation * (1 - self.excitation)
        log(self.name, ": val_tmp=", val_tmp, type(val_tmp))
        val_next = alpha * err * val_tmp
        log(self.name, ": val_next=", val_next, type(val_next))

        # adjust bias
        self.b += val_next
        log(self.name, ": b=")
        log(self.b)

        # adjust weights
        delta_w = np.outer(val_next, self.previous.get_excitation())
        log(type(delta_w), self.name, ": delta_w=")
        log(delta_w)
        self.w += delta_w
        log(self.name, ": w=")
        log(self.w)


class NeuronNet(object):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, random_seed):
        log("NeuronNet.initialize: input_layer_size =", input_layer_size,
            ", hidden_layer_size =", hidden_layer_size,
            ", output_layer_size =", output_layer_size,
            ", random_seed =", random_seed)

        expected = []
        for i in range(output_layer_size):
            x = [0.0] * output_layer_size
            x[i] = 1.0
            expected.append(x)
        self.expected = np.asarray(expected)
        log("Expectation matrix = ", expected)

        np.random.seed(random_seed)
        self.input_layer = Layer("input layer", input_layer_size)
        self.hidden_layer = Layer("hidden layer", hidden_layer_size)
        self.output_layer = Layer("output layer", output_layer_size)
        if hidden_layer_size > 0:
            self.hidden_layer.connect(self.input_layer)
            self.output_layer.connect(self.hidden_layer)
        else:
            self.output_layer.connect(self.input_layer)

    def train(self, all_pattern, alpha):
        num_pattern = len(all_pattern)
        log("NeuronNet.train: number of pattern =", num_pattern,
            ", alpha =", alpha)

        error_epoch = 0
        for i in range(num_pattern):
            # Select and apply training pattern
            output_index, data = all_pattern[i % num_pattern]
            self.input_layer.set(data)

            # Forward propagation
            if self.hidden_layer.size > 0:
                self.hidden_layer.forward()
            self.output_layer.forward()

            # Calculate error
            exp = self.expected[int(output_index)]
            error = exp - self.output_layer.get_excitation()
            log("error", error, type(error))
            error_total = (error ** 2).sum()
            log("error_total =", error_total)
            error_epoch += error_total

            # Perform back propagation
            self.output_layer.backward(error, alpha)
            if self.hidden_layer.size > 0:
                error = self.output_layer.get_error()
                self.hidden_layer.backward(error, alpha)

        return error_epoch / num_pattern

    def run(self, pattern_data):
        self.input_layer.set(pattern_data)
        if self.hidden_layer.size > 0:
            self.hidden_layer.forward()
        self.output_layer.forward()
        return self.hidden_layer.get_excitation(), self.output_layer.get_excitation()

    def test(self, pattern):
        output_index, data = pattern
        hidden_excitation, output_excitation = self.run(data)
        max_excitation = -100
        for i in range(len(output_excitation)):
            if output_excitation[i] > max_excitation:
                max_excitation = output_excitation[i]
                index_max_excitation = i

        index_expected = int(output_index)
        if index_max_excitation == index_expected:
            return True
        else:
            return False
