import monkdata as md
import dtree as dt
import drawtree_qt5 as dr

# assignment 1
print("\n----------ASSIGNMENT 1----------\n")
print("Entropy MONK-1 training dataset:",  dt.entropy(md.monk1))
print("Entropy MONK-2 training dataset:", dt.entropy(md.monk2))
print("Entropy MONK-3 training dataset:", dt.entropy(md.monk3))
print()

# assignment 3
print("\n----------ASSIGNMENT 3----------\n")
print("Information gain MONK-1")
gain = dict(zip(md.attributes, [dt.averageGain(md.monk1, a) for a in md.attributes]))
for a, g in gain.items():
    print("Information gain ", a, ": ", g, sep="")
selected = max(gain, key=gain.get)
print("Best attribute for split:", selected)
print()

print("Information gain MONK-2")
gain = dict(zip(md.attributes, [dt.averageGain(md.monk2, a) for a in md.attributes]))
for a, g in gain.items():
    print("Information gain ", a, ": ", g, sep="")
print("Best attribute for split:", max(gain, key=gain.get))
print()

print("Information gain MONK-3")
gain = dict(zip(md.attributes, [dt.averageGain(md.monk3, a) for a in md.attributes]))
for a, g in gain.items():
    print("Information gain ", a, ": ", g, sep="")
print("Best attribute for split:", max(gain, key=gain.get))
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
        gain = dict(zip(attributes_left, [dt.averageGain(subset, a) for a in attributes_left]))
        for a, g in gain.items():
            print("Information gain ", a, ": ", g, sep="")
        best = max(gain, key=gain.get)
        print("Best attribute for split:", best)
        for v2 in best.values:
            print(best, "=", v2, "->", dt.mostCommon(dt.select(subset, best, v2)))
    print()
dr.drawTree(dt.buildTree(md.monk1, md.attributes, 2))

# assignment 5
print("\n----------ASSIGNMENT 5----------\n")
t = dt.buildTree(md.monk1, md.attributes)
print("Performance MONK-1 training dataset:", dt.check(t, md.monk1))
print("Performance MONK-1 test dataset:", dt.check(t, md.monk1test))
t = dt.buildTree(md.monk2, md.attributes)
print("Performance MONK-2 training dataset:", dt.check(t, md.monk2))
print("Performance MONK-2 test dataset:", dt.check(t, md.monk2test))
t = dt.buildTree(md.monk3, md.attributes)
print("Performance MONK-3 training dataset:", dt.check(t, md.monk3))
print("Performance MONK-3 test dataset:", dt.check(t, md.monk3test))

# assignment 7
