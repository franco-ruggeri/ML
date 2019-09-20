import monkdata as m
import dtree as dt
import drawtree_qt5 as draw5
import random
import matplotlib.pyplot as plt
import numpy as np


def bestPruning(tree):
    stop = False
    previousTree = tree

    while not stop:
        AP = dt.allPruned(previousTree)
        if not AP:
            stop = True
        else:
            t_max = AP[0]
            v_max = dt.check(t_max, monk1val)
            for tree in AP:
                tree_val = dt.check(tree, monk1val)
                if tree_val > v_max and tree_val >= validation1:
                    t_max = tree
                    v_max = tree_val

            if t_max == AP[0] and dt.check(t_max, monk1val) < validation1:
                stop = True
            else:
                previousTree = t_max

    return previousTree


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


m1entropy = dt.entropy(m.monk1)
m2entropy = dt.entropy(m.monk2)
m3entropy = dt.entropy(m.monk3)

print("Entropy of MONK-1: ", m1entropy)
print("Entropy of MONK-2: ", m2entropy)
print("Entropy of MONK-3: ", m3entropy)

IGm1a1 = dt.averageGain(m.monk1, m.attributes[0])
IGm1a2 = dt.averageGain(m.monk1, m.attributes[1])
IGm1a3 = dt.averageGain(m.monk1, m.attributes[2])
IGm1a4 = dt.averageGain(m.monk1, m.attributes[3])
IGm1a5 = dt.averageGain(m.monk1, m.attributes[4])

IGm2a1 = dt.averageGain(m.monk2, m.attributes[0])
IGm2a2 = dt.averageGain(m.monk2, m.attributes[1])
IGm2a3 = dt.averageGain(m.monk2, m.attributes[2])
IGm2a4 = dt.averageGain(m.monk2, m.attributes[3])
IGm2a5 = dt.averageGain(m.monk2, m.attributes[4])

IGm3a1 = dt.averageGain(m.monk3, m.attributes[0])
IGm3a2 = dt.averageGain(m.monk3, m.attributes[1])
IGm3a3 = dt.averageGain(m.monk3, m.attributes[2])
IGm3a4 = dt.averageGain(m.monk3, m.attributes[3])
IGm3a5 = dt.averageGain(m.monk3, m.attributes[4])

print("MONK-1 attributes information Gain")
print(round(IGm1a1, 4), "\t", round(IGm1a2, 4), "\t", round(IGm1a3, 4), "\t", round(IGm1a4, 4), "\t", round(IGm1a5, 4))
print("MONK-2 attributes information Gain")
print(round(IGm2a1, 4), "\t", round(IGm2a2, 4), "\t", round(IGm2a3, 4), "\t", round(IGm2a4, 4), "\t", round(IGm2a5, 4))
print("MONK-3 attributes information Gain")
print(round(IGm3a1, 4), "\t", round(IGm3a2, 4), "\t", round(IGm3a3, 4), "\t", round(IGm3a4, 4), "\t", round(IGm3a5, 4))

t1 = dt.buildTree(m.monk1, m.attributes)
# draw5.drawTree(t1)
print("Train dataset error M1: ", dt.check(t1, m.monk1))
print("Test dataset error M1: ", dt.check(t1, m.monk1test))

t2 = dt.buildTree(m.monk2, m.attributes)
# draw5.drawTree(t2)
print("Train dataset error M2: ", dt.check(t2, m.monk2))
print("Test dataset error M2: ", dt.check(t2, m.monk2test))

t3 = dt.buildTree(m.monk3, m.attributes)
# draw5.drawTree(t)
print("Train dataset error M3: ", dt.check(t3, m.monk3))
print("Test dataset error M3: ", dt.check(t3, m.monk3test))

fractionArray = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
errorMonk1Array = []
errorArray = []

for fraction in fractionArray:
    for i in range(0, 500):
        monk1train, monk1val = partition(m.monk1, fraction)
        t1split = dt.buildTree(monk1train, m.attributes)
        validation1 = dt.check(t1split, monk1val)
        bestT1 = bestPruning(t1split)
        error = 1 - dt.check(bestT1, m.monk1test)
        errorArray.append(error)
        # print(dt.check(bestT1, monk1val))
        # print(validation1)
        # draw5.drawTree(t1split)
        # draw5.drawTree(bestT1)
    errorMonk1Array.append(np.mean(errorArray))
    errorArray.clear()


# Set chart title.
plt.title("MONK1 - Error test set with defined fraction for training\n(average on 500 runs)")

# Set x, y label text.
plt.xlabel("Fraction")
plt.ylabel("Error")
plt.scatter(fractionArray, errorMonk1Array, s=5)
plt.plot(fractionArray, errorMonk1Array)
plt.show()

# monk2train, monk2val = partition(m.monk2, 0.6)
# t2split = dt.buildTree(monk2train, m.attributes)
# validation2 = dt.check(t2split, monk2val)
# bestT2 = bestPruning(t2split)
# print(dt.check(bestT2, monk2val))
# print(validation2)
# draw5.drawTree(bestT2)

errorMonk3Array = []

for fraction in fractionArray:
    for i in range(0, 500):
        monk3train, monk3val = partition(m.monk3, fraction)
        t3split = dt.buildTree(monk3train, m.attributes)
        validation3 = dt.check(t3split, monk3val)
        bestT3 = bestPruning(t3split)
        error = 1 - dt.check(bestT3, m.monk3test)
        errorArray.append(error)
        # print(dt.check(bestT1, monk1val))
        # print(validation1)
        # draw5.drawTree(t1split)
        # draw5.drawTree(bestT1)
    errorMonk3Array.append(np.mean(errorArray))
    errorArray.clear()

# print(dt.check(bestT3, monk3val))
# print(validation3)
# draw5.drawTree(bestT3)

# Set chart title.
plt.title("MONK3 - Error test set with defined fraction for training\n(average on 500 runs)")

# Set x, y label text.
plt.xlabel("Fraction")
plt.ylabel("Error")
plt.scatter(fractionArray, errorMonk3Array, s=10)
plt.plot(fractionArray, errorMonk3Array)
plt.show()


