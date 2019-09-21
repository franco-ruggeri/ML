import monkdata as md
import dtree as dt
import drawtree_qt5 as dr
import random as r
import matplotlib.pyplot as p

# assignment 1
print("\n----------ASSIGNMENT 1----------\n")
print("Entropy MONK-1 training dataset:",  dt.entropy(md.monk1))
print("Entropy MONK-2 training dataset:", dt.entropy(md.monk2))
print("Entropy MONK-3 training dataset:", dt.entropy(md.monk3))
print()


# assignment 3
print("\n----------ASSIGNMENT 3----------\n")
gains = dict(zip(md.attributes, [dt.averageGain(md.monk1, a) for a in md.attributes]))
print("Information gains MONK-1:", gains)
selected = max(gains, key=gains.get)
print("Best attribute for split:", selected)
print()

gains = dict(zip(md.attributes, [dt.averageGain(md.monk2, a) for a in md.attributes]))
print("Information gains MONK-2:", gains)
print("Best attribute for split:", max(gains, key=gains.get))
print()

gains = dict(zip(md.attributes, [dt.averageGain(md.monk3, a) for a in md.attributes]))
print("Information gains MONK-3:", gains)
print("Best attribute for split:", max(gains, key=gains.get))
print()


# building tree
print("\n----------DECISION TREE MONK-1 DEPTH 2----------\n")
for v in selected.values:
    print(selected, "=", v)
    subset = dt.select(md.monk1, selected, v)
    if dt.allPositive(subset) or dt.allNegative(subset):
        print(selected, "=", v, "->", dt.mostCommon(subset))
    else:
        attributes_left = [a for a in md.attributes if a != selected]
        gains = dict(zip(attributes_left, [dt.averageGain(subset, a) for a in attributes_left]))
        print("Information gains:", gains)
        best = max(gains, key=gains.get)
        print("Best attribute for split:", best)
        for v2 in best.values:
            print(best, "=", v2, "->", dt.mostCommon(dt.select(subset, best, v2)))
    print()
# dr.drawTree(dt.buildTree(md.monk1, md.attributes, 2))
print(dt.buildTree(md.monk1, md.attributes, 2))


# assignment 5
print("\n----------ASSIGNMENT 5----------\n")
tree = dt.buildTree(md.monk1, md.attributes)
print("Prediction error MONK-1 training dataset:", 1 - dt.check(tree, md.monk1))
print("Prediction error MONK-1 test dataset:", 1 - dt.check(tree, md.monk1test))
tree = dt.buildTree(md.monk2, md.attributes)
print("Prediction error MONK-2 training dataset:", 1 - dt.check(tree, md.monk2))
print("Prediction error MONK-2 test dataset:", 1 - dt.check(tree, md.monk2test))
tree = dt.buildTree(md.monk3, md.attributes)
print("Prediction error MONK-3 training dataset:", 1 - dt.check(tree, md.monk3))
print("Prediction error MONK-3 test dataset:", 1 - dt.check(tree, md.monk3test))


# assignment 7
def partition(data, fraction):
    "Partition data (dataset) in two subsets whose sizes depend on fraction"
    ldata = list(data)
    r.shuffle(ldata)
    break_point = int(len(ldata) * fraction)
    return ldata[:break_point], ldata[break_point:]

def reduced_error_pruning(data, attributes, fraction):
    "Generate the pruned tree according to the reduced error pruning"
    training, validation = partition(data, fraction)
    best_tree = dt.buildTree(training, attributes)
    best_performance = dt.check(best_tree, validation)
    complete = False
    while not complete:
        alternatives = [(t, dt.check(t, validation)) for t in dt.allPruned(best_tree)]
        t, p = max(alternatives, key=lambda x: x[1])
        if p >= best_performance:   # greater OR EQUAL!!
            best_tree, best_performance = t, p
        else:
            complete = True
    return best_tree

print("\n----------ASSIGNMENT 7----------\n")
N = 100

# MONK-1
mean_errors = []
spread_errors = []
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for f in fractions:
    errors = []
    for i in range(N):
        t = reduced_error_pruning(md.monk1, md.attributes, f)
        errors.append(1 - dt.check(t, md.monk1test))
    mean_errors.append(sum(errors) / N)
    spread_errors.append(max(errors) - min(errors))
p.plot(fractions, mean_errors, 's')
p.errorbar(fractions, mean_errors, spread_errors)
p.xlabel("Fraction")
p.ylabel("Error")
p.title("MONK-1 - Errors of pruned trees\naverage on {} runs".format(N))
p.show()

# MONK-3
mean_errors.clear()
spread_errors.clear()
for f in fractions:
    errors = []
    for i in range(N):
        t = reduced_error_pruning(md.monk3, md.attributes, f)
        errors.append(1 - dt.check(t, md.monk3test))
    mean_errors.append(sum(errors) / N)
    spread_errors.append(max(errors) - min(errors))
p.plot(fractions, mean_errors, 's')
p.errorbar(fractions, mean_errors, spread_errors)
p.xlabel("Fraction")
p.ylabel("Error")
p.title("MONK-2 - Errors of pruned trees\naverage on {} runs".format(N))
p.show()