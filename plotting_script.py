import numpy as np, matplotlib.pyplot as plt

x = np.load("data/omniglot_x_train.npy")
y = np.load("data/omniglot_y_train.npy")
fig, ax = plt.subplots(3, 10, figsize=(12, 4))
for r, cls in enumerate(np.unique(y)[[0, 400, 900]]):
    imgs = x[y == cls]
    for j in range(10):
        ax[r, j].imshow(imgs[j].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        ax[r, j].axis("off")
plt.tight_layout()
plt.savefig("check.png", dpi=110)
