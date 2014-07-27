"""
Back-propagation neural network implementation, using numpy

Adapted from http://stackoverflow.com/a/3143318
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

  def __init__(self, n_in, n_out, *hidden):
    # Initialize size of each layer
    # One extra input for biases
    self.layers = array([node_count + 1 for node_count in (n_in,) + hidden] + [n_out])
    self.n_layers = len(self.layers)

    # Initialize activation values for each node
    # Could use numpy array, but they're updated sequentially anyway
    self.activations = [ones(node_count) for node_count in self.layers]
    
    # Initialize weights between levels
    self.weights = [random.uniform(-0.5, 0.5, [i, j]) for i,j in zip(self.layers[1:], self.layers[:-1])]

    # Initialize previous change in weights, for momentum
    self.past_change = [zeros([i, j]) for i,j in zip(self.layers[1:], self.layers[:-1])]

  def activate(self, inputs):
    """Activate all nodes, and output last layer"""
    # Append bias term
    self.activations[0] = append(inputs, 1.0)

    # Activate each hidden layer, using sigmoid function
    for i in xrange(self.n_layers - 2):
      self.activations[i + 1] = append(sigmoid(self.weights[i].dot(self.activations[i])), 1.0)
    
    # Activate output layer, don't use sigmoid function
    # Has the advantage of simulating a perceptron when there are no hidden layers
    self.activations[-1] = self.weights[-1].dot(self.activations[-2])

    # Return a copy of the output layer
    return self.activations[-1][:]

  def backPropagate(self, targets, a, M):
    """Update weights assuming neural network is activated to achieve targets""" 

    current_change = [zeros([i,j]) for i,j in zip(self.layers[1:], self.layers[:-1])]
    
    error = self.activations[-1] - targets
    # The actual cost function being minimized
    err = 0.5 * linalg.norm(error)

    deltas = error
    current_change[-1] = outer(error, self.activations[-2])

    for i in reversed(xrange(self.n_layers - 2)):
      error = self.weights[i + 1].T.dot(deltas)
      deltas = dsigmoid(self.activations[i + 1]) * error

      current_change[i] = outer(deltas, self.activations[i])

    for i in xrange(self.n_layers - 1):
      self.weights[i] -= M*self.past_change[i] + a*current_change[i]
      self.past_change[i] = current_change[i]

    return err

  def train(self, data, targets, num_epochs=1000, a=0.01, M=0.002):
    """Trains the neural network"""
    #TODO: normalize inputs and outputs. Definitely need to do this
    error = 0.0
    for i in xrange(num_epochs):
      error = 0.0
      for x,y in zip(data, targets):
        self.activate(x)
        error += self.backPropagate(y, a, M)
      if i % 100 == 0:
        print "Iteration: %s, Error: %s" % (i,error)
    return error