import numpy as np


class Activation:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        #Update self.output
        self.output = None


class Activation_ReLU(Activation):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax(Activation):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Activation_ELU(Activation):
    def forward(self, inputs):
        alpha = 1
        self.output = inputs if inputs >= 0 else alpha*(np.exp(inputs) - 1)
