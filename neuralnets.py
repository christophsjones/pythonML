"""
Back-propagation neural network, using numpy

Heavily modified from http://stackoverflow.com/a/3143318
"""

from numpy import *

# Sigmoid function used in this neural network
def sigmoid(xs):
  """Compute sigmoid at each x in vector"""
  return 1.0 / (1.0 + exp(-xs))

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
    self.bias_correct = ~identity(self.n_layers - 1, dtype=bool)[-1]
    self.weights = [random.uniform(-0.5, 0.5, [i, j]) for i,j in zip(self.layers[1:] - self.bias_correct, self.layers[:-1])]
    self.past_change = [zeros([i, j]) for i,j in zip(self.layers[1:] - self.bias_correct, self.layers[:-1])]

    self.input_scale = (ones(n_in), zeros(n_in))
    self.output_scale = (ones(n_out), zeros(n_out))

  def activate(self, inputs):
    """Activate all nodes, and output last layer"""
    inputs = (inputs - self.input_scale[1]).astype(float) / self.input_scale[0]

    ans = self._activate(inputs) * self.output_scale[0] + self.output_scale[1]
    # Return a copy of the activation value for modification
    return ans[:]

  def _activate(self, inputs):
    """Internal activation, does not scale input/output"""
    # Append bias term
    self.activations[0] = append(inputs, 1.0)

    # Activate each hidden layer, using sigmoid function
    for i in xrange(self.n_layers - 2):
      self.activations[i + 1] = append(sigmoid(self.weights[i].dot(self.activations[i])), 1.0)
    
    # Activate output layer, don't use sigmoid function
    # Has the advantage of simulating a perceptron when there are no hidden layers
    self.activations[-1] = self.weights[-1].dot(self.activations[-2])

    return self.activations[-1]

  def backPropagate(self, targets, a, M):
    """Update weights assuming neural network is activated to achieve targets""" 

    current_change = [zeros([i,j]) for i,j in zip(self.layers[1:] - self.bias_correct, self.layers[:-1])]
    
    error = self.activations[-1] - targets
    # The actual cost function being minimized
    err = 0.5 * dot(error,error.conj())

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
    
    # Scale inputs to [-1,1]
    # Could use log scale if there are outliers
    u,v = amax(data, 0), amin(data, 0)
    w = u - v
    w[w == 0.0] = 2.0
    self.input_scale = (0.5 * w, 0.5 * (u + v))
    data = (data - self.input_scale[1]).astype(float) / self.input_scale[0]
    print "Input scale:", self.input_scale

    # If amax == amin, GG
    u,v = amax(targets, 0), amin(targets, 0)
    self.output_scale = (0.5 * (u - v), 0.5 * (u + v))
    targets = (targets - self.output_scale[1]).astype(float) / self.output_scale[0]
    print "Output scale:", self.output_scale

    error = 0.0
    for i in xrange(num_epochs):
      error = 0.0
      for x,y in zip(data, targets):
        self.activate(x)
        error += self.backPropagate(y, a, M)
      if verbose and i % (num_epochs / 10) == 0:
        print "Iteration: %s of %s, Error: %s" % (i,num_epochs,error)
    return error