# Numerical check on Problem Set 6-2 and 6-3

import matplotlib.pyplot as plt

from numpy import *
from sklearn.decomposition import PCA

def ellipse(a, t, n):
    # generate a list of n random points, some contained within a rotated ellipse
    # rotation: t
    e = 2 * random.rand(n,2) - 1.0
    e = array(filter(lambda p: (p**2).dot(a) <= 1, e))

    # rotate points in e
    rotmat = array([[cos(t), sin(t)], [-sin(t), cos(t)]])
    return e.dot(rotmat)

def scatterplot(e):
    # draw scatter plot of points in e
    x = []
    y = []
    for (r, s) in e:
        x.append(r)
        y.append(s)
    plt.scatter(x, y)
    plt.show()

# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
def principal(e):
    # perform principal component analysis on e
    pca = PCA()
    pca.fit(e)
    print pca.components_
    print pca.explained_variance_ratio_, "   explained variance ratio"

def variance(e, vec):
    #project along vec
    vals = e.dot(vec) / linalg.norm(vec)
    print "Variance in", vec, "direction:", var(vals)

def main():
    axis = array([25.0, 1.0])

    # generate points in the first ellipse
    #e  = ellipse(0.20, 1.00, math.pi/4.0, 200)
    #scatterplot(e)
    #print "Problem 6-2 Analogy - PCA components"
    #principal(e)

    # generate points in the second ellipse
    # e = append(ellipse(axis, pi/4.0, 1500), ellipse(axis[::-1], pi/4.0, 1500), 0)
    e = 2 * random.rand(10000, 2) - 1.0
    rotmat = 2**0.5 * array([[1, 1], [-1,1]])
    e = array(filter(lambda p: ((rotmat.dot(p))**2).dot(axis) <= 1 or ((rotmat.dot(p))**2).dot(axis[::-1]) <= 1, e))

    scatterplot(e)
    print "Problem 6-3 Analogy - variance"
    variance(e,array([1,0]))
    variance(e,array([0,1]))
    variance(e,array([1,1]))
    variance(e,array([1,-1]))
    print "Problem 6-3 Analogy - PCA components"
    principal(e)


print __name__
if __name__ == "__main__":
  pass
