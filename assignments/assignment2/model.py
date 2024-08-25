import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer = ReLULayer()
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        params = self.params()
        for _, param in params.items():
            param.grad = 0

        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        #forward
        l1_out = self.first_layer.forward(X)
        relu_out = self.relu_layer.forward(l1_out)
        l2_out = self.second_layer.forward(relu_out)
        loss, d_pred = softmax_with_cross_entropy(l2_out, y)
        
        #backward
        grad_l2 = self.second_layer.backward(d_pred)
        grad_relu   = self.relu_layer.backward(grad_l2)
        grad_l1  = self.first_layer.backward(grad_relu)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for _, param in params.items():
            l2_loss, l2_grad = l2_regularization(param.value, self.reg)
            loss += l2_loss
            param.grad += l2_grad
            
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        l1_out = self.first_layer.forward(X)
        relu_out = self.relu_layer.forward(l1_out)
        l2_out = self.second_layer.forward(relu_out)
        
        return np.argmax(l2_out, axis=1)

    def params(self):
        result = {
            'W1': self.first_layer.params()['W'],
            'B1': self.first_layer.params()['B'],
            'W2': self.second_layer.params()['W'],
            'B2': self.second_layer.params()['B']
        }
        return result
