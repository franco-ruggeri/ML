import numpy
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

################
# Init #########
################

# init seed (debugging)
# numpy.random.seed(100)

# 2 clusters of class A (+1)
classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
     numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))

# 1 cluster of class B (-1)
classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

# training set (data samples and labels)
inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate(
    (numpy.ones(classA.shape[0]),
     -numpy.ones(classB.shape[0])))
N = inputs.shape[0]

# shuffle training set
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute]
targets = targets[permute]


#####################
# Build SVM #########
#####################

# functions (they use global variables defined below)
def linear_kernel(x, y):
    return numpy.dot(x, y)


def polynomial_kernel(x, y, p):
    return (numpy.dot(x, y) + 1) ** p


def radial_kernel(x, y, sigma):
    return math.exp(- numpy.linalg.norm(numpy.array(x) - y)**2 / (2 * sigma**2))


def objective(a):
    return 1/2 * numpy.dot(a, numpy.dot(a, P)) - numpy.sum(a)


def zerofun(a):
    return numpy.dot(a, targets)


def ind(y):
    return numpy.sum([x['alpha'] * x['target'] * K(y, x['point']) for x in support_vectors]) - b


# choose kernel
K = linear_kernel
# p = 2
# K = lambda x, y: polynomial_kernel(x, y, p)
# sigma = 1e-1
# K = lambda x, y: radial_kernel(x, y, sigma)

# soft margins
C = None

# matrix to avoid computing the same quantities a lot of times
P = [[targets[i] * targets[j] * K(inputs[i], inputs[j]) for j in range(N)] for i in range(N)]

# optimization (minimization of dual problem)
ret = minimize(objective, numpy.zeros(N), bounds=[(0, C) for x in range(N)], constraints={'type': 'eq', 'fun': zerofun})
if not ret['success']:
    print('No solution found by minimize()')
    exit(-1)
alpha = ret['x']

# extract non-zero alpha
support_vectors = [{'point': inputs[i], 'alpha': alpha[i], 'target': targets[i]}
                   for i in range(N) if abs(alpha[i]) > 1e-5]
if len(support_vectors) < 1:
    print('No support vectors')
    exit(-1)

# compute b
s = support_vectors[0]
b = numpy.sum([x['alpha'] * x['target'] * K(s['point'], x['point']) for x in support_vectors]) - s['target']


####################
# Analysis #########
####################

# plot training set
plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')

# plot support vectors
plt.plot([x['point'][0] for x in support_vectors],
         [x['point'][1] for x in support_vectors],
         'yo')

# plot decision boundary and margin
xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)
grid = numpy.array([[ind([x, y]) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1),
            linestyles=('dashed', 'solid', 'dashed'))

# configure and show plot
plt.axis([-2, 2, -2, 2])
plt.show()

# save plot
# plt.savefig('svmplot.pdf')
