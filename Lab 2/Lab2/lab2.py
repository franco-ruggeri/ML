import numpy
import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import animation

################
# Init #########
################

# init seed (debugging)
numpy.random.seed(1)

# 2 clusters of class A (+1)
# classA = numpy.concatenate(
#     (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
#      numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))

#
classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [0.5, 0.5],
     numpy.random.randn(10, 2) * 0.2 + [-0.5, 0.5]))

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
numpy.random.seed(1)
random.shuffle(permute)
inputs = inputs[permute]
targets = targets[permute]

fig = plt.figure()

# plt.plot([p[0] for p in classA],
#          [p[1] for p in classA],
#          'b.')
# plt.plot([p[0] for p in classB],
#          [p[1] for p in classB],
#          'r.')

plt.title("")
# plt.show()

# ax = plt.axis([-2, 2, -2, 2])
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)


#####################
# Build SVM #########
#####################

# functions (they use global variables defined below)
def linear_kernel(x, y):
    return numpy.dot(x, y)


def polynomial_kernel(x, y, p=3):
    return (numpy.dot(x, y) + 1) ** p


def radial_kernel(x, y, sigma=0.5):
    return math.exp(- numpy.linalg.norm(numpy.array(x) - y) ** 2 / (2 * sigma ** 2))


def objective(a):
    return 1 / 2 * numpy.dot(a, numpy.dot(P, a)) - numpy.sum(a)


def zerofun(a):
    return numpy.dot(a, targets)


def ind(y, support_vectors, b):
    return numpy.sum([x['alpha'] * x['target'] * K(y, x['point']) for x in support_vectors]) - b


# choose kernel
K = radial_kernel
# p = 2
# K = lambda x, y: polynomial_kernel(x, y, p)
# sigma = 1e-1
# K = lambda x, y: radial_kernel(x, y, sigma)

# matrix to avoid computing the same quantities a lot of times
P = [[targets[i] * targets[j] * K(inputs[i], inputs[j]) for j in range(N)] for i in range(N)]


def animate(i):
    # soft margins
    C = 0.15 + i * 0.01
    # optimization (minimization of dual problem)
    ret = minimize(objective, numpy.zeros(N), bounds=[(0, C) for x in range(N)],
                   constraints={'type': 'eq', 'fun': zerofun})
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

    # # plot training set
    # plt.plot([p[0] for p in classA],
    #          [p[1] for p in classA],
    #          'b.')
    # plt.plot([p[0] for p in classB],
    #          [p[1] for p in classB],
    #          'r.')

    # plot support vectors

    # plot decision boundary and margin
    xgrid = numpy.linspace(-5, 5)
    ygrid = numpy.linspace(-4, 4)
    grid = numpy.array([[ind([x, y], support_vectors, b) for x in xgrid] for y in ygrid])
    # line.set_data(xgrid, ygrid, grid,
    #             (-1.0, 0.0, 1.0),
    #             colors=('red', 'black', 'blue'),
    #             linewidths=(1, 3, 1),
    #             linestyles=('dashed', 'solid', 'dashed'))

    ax.clear()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    title = "C = " + str("{0:.2f}".format(round(C, 3)))
    ax.set_title(title)
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.')
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')
    # plt.plot([x['point'][0] for x in support_vectors],
    #          [x['point'][1] for x in support_vectors],
    #          'yo')
    cont = plt.contour(xgrid, ygrid, grid,
                       (-1.0, 0.0, 1.0),
                       colors=('red', 'black', 'blue'),
                       linewidths=(1, 3, 1),
                       linestyles=('dashed', 'solid', 'dashed'))
    return cont


def init():
    line.set_data([], [])
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=400, interval=1, blit=False)

anim.save('radial_0_5_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

# configure and show plot
# plt.axis([-2, 2, -2, 2])
# plt.show()

# save plot
# plt.savefig('svmplot.pdf')
