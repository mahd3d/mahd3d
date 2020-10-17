from matplotlib import pyplot as plt


def matplot() -> None:
    """Deprecated."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100
    for xs, ys, zs in zip(x, y, z):
        ax.scatter(xs, ys, zs, marker='o', c=(255, 255, 255, 0))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
