"""
Decision trees for regression, using numpy

Based on the ID3 algorithm
"""

from numpy import *

def mutual_information(data, targets):
  """Return the information gain from each attribute

  Kullback-Leibler divergence of prior distribution from posterior distribution, for each attribute
  """
  #TODO
  return zeros(len(data[0]))

class TreeNode(object):
  def __init__(self, l=None, r=None, a=None, t=None, v=None):
    self.leftChild = l
    self.rightChild = r
    self.attribute = a
    self.threshold = t
    
    self.value = v

  def activate(self, inputs):
    if self.value is not None:
      return self.value

    if inputs[self.attribute] <= self.threshold:
      return self.leftChild.activate(inputs)
    else:
      return self.rightChild.activate(inputs)

class DT(object):
  def __init__(self, max_depth=3):
    self.tree = TreeNode()
    self.max_depth = max_depth

  def activate(self, inputs):
    return self.tree.activate(inputs)

  def buildTree(self, data, targets, depth, verbose):
    if not depth or array_equiv(targets, targets[0]):
      return TreeNode(v=targets.mean(0))
    splitAttribute = argmax(mutual_information(data, targets))
    threshold = data.T[splitAttribute].mean()

    if verbose:
      print "Splitting on feature", splitAttribute, "with threshold", threshold

    partition = data.T[splitAttribute] <= threshold
    leftTree = self.buildTree(data[partition], targets[partition], depth - 1, verbose)
    rightTree = self.buildTree(data[~partition], targets[~partition], depth - 1, verbose)

    return TreeNode(l=leftTree, r=rightTree, a=splitAttribute, t=threshold)

  def train(self, data, targets, verbose=True):
    self.tree = self.buildTree(data, targets, self.max_depth, verbose)
    
    error = 0.0
    for x,y in zip(data, targets):
      err = self.activate(x) - y
      error += dot(err, err)      

    return error / len(data)
