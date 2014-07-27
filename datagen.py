from numpy import *

def linear(m=200, n=10):
  xs = random.rand(m, n)
  coeff = 2 * random.rand(n) - 1
  ys = xs.dot(coeff)
  return {"data" : xs, "target" : ys, "coeff" : coeff, "dim" : (n,1)}

def linear_approx(m=200, n=10, s=0.05):
  xs = random.rand(m,n)
  coeff = 2 * random.rand(n) - 1
  ys = xs.dot(coeff) + s * random.randn(m)
  return {"data" : xs, "target" : ys, "coeff" : coeff, "dim" : (n,1)}

