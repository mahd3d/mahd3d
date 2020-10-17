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


class Line:
    p1 = (0, 0, 0)
    x1 = 0
    y1 = 0
    z1 = 0
    p2 = (1, 1, 1)
    x2 = 0
    y2 = 0
    z2 = 0
    m = (1, 1, 1)

    def equation(self):
        return f"= f{self.p1} + t {self.p2}"


class Sphere:
    p = (0.5, 0.5, 0.5)
    x = 0.5
    y = 0.5
    z = 0.5
    r = 0.5
    d = 1.0

    def equation(self):
        return f"(x-{self.x})²+(y-{self.y})²+(y-{self.y})² = {self.r ** 2}"
