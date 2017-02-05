import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden, output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        # dimension here is like (output_layer X input_layer)
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                         (self.output_nodes, self.hidden_nodes))

        self.lr = learning_rate

        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1/(1+np.exp(-x))

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        ###Implement the forward pass here
        ##Hidden Layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        ##Output Layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs # f(x) = x is activation of output layer

        ###Implement the backward pass here
        ##Output error
        #Output layer error is the difference b/w desired target and actual output
        output_errors = targets - final_outputs

        ##Backpropagated error
        #hidden layer gradients
        hidden_grad = hidden_outputs*(1-hidden_outputs)
        #errors propagated to the hidden layer
        hidden_errors = np.dot(output_errors, self.weights_hidden_to_output).T

        #Update weights of the layers
        #Update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T)
        #Update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * np.dot(hidden_errors*hidden_grad, inputs.T)

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        ###Implement the forward pass
        # dim =>  (output_layer x input_layer) X (input X 1)
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        ###Implement the output layer
        # dim => (output_layer X input_layer) X (input X 1)
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        # Since Activation function of final layer is f(x) = x => f'(x) = 1
        final_outputs = final_inputs

    def MSE(y, Y):
        return np.mean((y - Y) ** 2)