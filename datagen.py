from numpy import *

def linear(m=200, n=10):
  xs = random.rand(m, n)
  coeff = 2 * random.rand(n) - 1
  ys = xs.dot(coeff)
  return {"data" : xs, "target" : ys, "dim" : (n,1), "coeff" : coeff}

def linear_approx(m=200, n=10, s=0.05):
  xs = random.rand(m,n)
  coeff = 2 * random.rand(n) - 1
  ys = xs.dot(coeff) + s * random.randn(m)
  return {"data" : xs, "target" : ys, "dim" : (n,1), "coeff" : coeff}

def fromfunc(f=lambda x,y: x**2 + y**2, m=200, n=2):
  xs = random.rand(m,n)
  ys = zeros(m)
  for i in xrange(m):
    ys[i] = f(xs[i])
  return {"data" : xs, "target" : ys, "dim" : (n,1)}

def fromfunc(f=lambda x,y: x**2 + y**2, m=200, n=2, s=0.05):
  xs = random.rand(m,n)
  ys = s * random.randn(m)
  for i in xrange(m):
    ys[i] += f(xs[i])
  return {"data" : xs, "target" : ys, "dim" : (n,1)}

