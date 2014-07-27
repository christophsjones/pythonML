"""
Back-propagation neural network, using numpy

Heavily adapted from http://stackoverflow.com/a/3143318
"""

from numpy import *

# Sigmoid function used in this neural network
def sigmoid(xs):
  """Compute sigmoid at each x in vector"""
  return 1.0 / (1 + exp(-xs))

def dsigmoid(ys):
  """Compute sigmoid derivative from each y in vector"""
  return ys * (1.0 - ys)

class NN(object):

  def __init__(self, n_in, *hidden):
    hidden, n_out = hidden[:-1], hidden[-1]
    # Initialize size of each layer
    # One extra input for biases
    self.layers = array([node_count + 1 for node_count in (n_in,) + hidden] + [n_out])
    self.n_layers = len(self.layers)

    # Initialize activation values for each node
    # Could use numpy array, but they're updated sequentially anyway
    self.activations = [ones(node_count) for node_count in self.layers]
    
    # Initialize weights between levels, and past changes for momentum
    self.bias_correct = ones(self.n_layers - 1) - identity(self.n_layers - 1)[-1]
    self.weights = [random.uniform(-0.5, 0.5, [i, j]) for i,j in zip(self.layers[1:] - self.bias_correct, self.layers[:-1])]
    self.past_change = [zeros([i, j]) for i,j in zip(self.layers[1:] - self.bias_correct, self.layers[:-1])]

    self.scale = (ones(n_in), ones(n_out))

  def activate(self, inputs):
    """Activate all nodes, and output last layer"""
    # Append bias term
    inputs = inputs.astype(float) / self.scale[0]
    self.activations[0] = append(inputs, 1.0)

    # Activate each hidden layer, using sigmoid function
    for i in xrange(self.n_layers - 2):
      self.activations[i + 1] = append(sigmoid(self.weights[i].dot(self.activations[i])), 1.0)
    
    # Activate output layer, don't use sigmoid function
    # Has the advantage of simulating a perceptron when there are no hidden layers
    self.activations[-1] = self.weights[-1].dot(self.activations[-2])

    # Return a copy of the output layer
    return self.activations[-1][:] * self.scale[1]

  def backPropagate(self, targets, a, M):
    """Update weights assuming neural network is activated to achieve targets""" 

    current_change = [zeros([i,j]) for i,j in zip(self.layers[1:] - self.bias_correct, self.layers[:-1])]
    
    error = self.activations[-1] - targets
    # The actual cost function being minimized
    err = 0.5 * linalg.norm(error)

    deltas = append(error, 0.0)
    current_change[-1] = outer(error, self.activations[-2])

    for i in reversed(xrange(self.n_layers - 2)):
      error = self.weights[i + 1].T.dot(deltas[:-1])
      deltas = dsigmoid(self.activations[i + 1]) * error

      current_change[i] = outer(deltas[:-1], self.activations[i])

    for i in xrange(self.n_layers - 1):
      self.weights[i] -= M*self.past_change[i] + a*current_change[i]
      self.past_change[i] = current_change[i]

    return err

  def train(self, data, targets, num_epochs=1000, a=0.01, M=0.002, verbose=True):
    """Trains the neural network"""
    
    # Scale inputs to [0,1]
    # Could use log scale if there are outliers
    self.scale = (amax(data, 0), amax(targets, 0))
    data = data.astype(float) / self.scale[0]
    targets = targets.astype(float) / self.scale[1]

    error = 0.0
    for i in xrange(num_epochs):
      error = 0.0
      for x,y in zip(data, targets):
        self.activate(x)
        error += self.backPropagate(y, a, M)
      if verbose and i % (num_epochs / 10) == 0:
        print "Iteration: %s of %s, Error: %s" % (i,num_epochs,error)
    return error