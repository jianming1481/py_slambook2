import numpy as np
import sys

def cubicInterpolate(p, x):
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))

def bicubicInterpolate(p, x, y):
    arr = np.zeros(4)
    for i in range(4):
        arr[i] = cubicInterpolate(p[i], y)
    return cubicInterpolate(arr, x)

def tricubicInterpolate(p, x, y, z):
    arr = np.zeros(4)
    for i in range(4):
        arr[i] = bicubicInterpolate(p[i], y, z)
    return cubicInterpolate(arr, x)

def nCubicInterpolate(n, p, coordinates):
    assert n > 0
    if n == 1:
        return cubicInterpolate(p, coordinates[0])
    else:
        arr = np.zeros(4)
        skip = 1 << (n - 1) * 2
        for i in range(4):
            arr[i] = nCubicInterpolate(n - 1, p[i * skip:], coordinates[1:])
        return cubicInterpolate(arr, coordinates[0])

if __name__ == "__main__":
    # Create array
    p = np.array([[1,3,3,4], [7,2,3,4], [1,6,3,6], [2,5,7,2]], dtype=float)

    # Interpolate
    print(bicubicInterpolate(p, 0.1, 0.2))

    # Or use the nCubicInterpolate function
    co = [0.1, 0.2]
    print(nCubicInterpolate(2, p.flatten(), co))
