import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier

# A 5x5 dataset with three classes:
# C on the outside, A in the inner ring, and B at the center.
X, y = [], []
for y2 in range(5):
    for x1 in range(5):
        if x1 == 2 and y2 == 2:
            label = 1  # B
        elif x1 in (1, 2, 3) and y2 in (1, 2, 3):
            label = 0  # A
        else:
            label = 2  # C
        X.append([x1, y2])
        y.append(label)

X = np.array(X)
y = np.array(y)

colors = ListedColormap(["#f2ad59", "#49b8a4", "#7da9e8"])
xx, yy = np.meshgrid(
    np.arange(-0.5, 4.51, 0.02),
    np.arange(-0.5, 4.51, 0.02),
)
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

for ax, depth in zip(axes, [2, 10]):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(X, y)

    # Predict across the plane, then color each rectangular decision region.
    regions = tree.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, regions, cmap=colors, alpha=0.5, levels=[-0.5, 0.5, 1.5, 2.5])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=colors, edgecolor="black", s=70)
    ax.set(title=f"max_depth={depth}", xlabel="x1", ylabel="x2")
    ax.set(xlim=(-0.5, 4.5), ylim=(-0.5, 4.5), xticks=range(5), yticks=range(5))
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

plt.show()
