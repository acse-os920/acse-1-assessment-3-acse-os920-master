import numpy as np
import copy
import timeit
import csv


def gauss(a, b):
    """
    Given two matrices, `a` and `b`, with `a` square, the determinant
    of `a` and a matrix `x` such that a*x = b are returned.
    If `b` is the identity, then `x` is the inverse of `a`.

    Parameters
    ----------
    a : np.array or list of lists
        'n x n' array
    b : np. array or list of lists
        'm x n' array

    Examples
    --------
    >> a = [[2, 0, -1], [0, 5, 6], [0, -1, 1]]
    >> b = [[2], [1], [2]]
    >> det, x = gauss(a, b)
    >> det
    22.0
    >> x
    [[1.5], [-1.0], [1.0]]
    >> A = [[1, 0, -1], [-2, 3, 0], [1, -3, 2]]
    >> I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >> Det, Ainv = gauss(A, I)
    >> Det
    3.0
    >> Ainv
    [[2.0, 1.0, 1.0],
    [1.3333333333333333, 1.0, 0.6666666666666666],
    [1.0, 1.0, 1.0]]

    Notes
    -----
    See https://en.wikipedia.org/wiki/Gaussian_elimination for further details.
    """
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    n = len(a)
    p = len(b[0])
    det = np.ones(1, dtype=np.float64)
    for i in range(n - 1):
        k = i
        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[k][i]):
                k = j
        if k != i:
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
            det = -det

        for j in range(i + 1, n):
            t = a[j][i]/a[i][i]
            for k in range(i + 1, n):
                a[j][k] -= t*a[i][k]
            for k in range(p):
                b[j][k] -= t*b[i][k]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            t = a[i][j]
            for k in range(p):
                b[i][k] -= t*b[j][k]
        t = 1/a[i][i]
        det *= a[i][i]
        for j in range(p):
            b[i][j] *= t
    return det, b


def gauss_np(a):

    det2 = np.linalg.det(a)

    return det2


def det(a):

    det3 = 0

    if(len(a) == 2):
        value = a[0][0]*a[1][1] - a[0][1]*a[1][0]
        return value

    for col in range(len(a)):

        for r in (a[:0] + a[1:]):

            ab = [r[:col] + r[col+1:]]

            if not ab:

                continue

            det3 = det3 + (-1)**(0 + col)*det(ab)*a[0][col]

    return det3


def rand_mat(n):

    return np.random.rand(n, n)


def time_mat(n):

    a = rand_mat(n)
    b = a.tolist()

    time_gauss = timeit.Timer(lambda: gauss(a, a)).timeit(number=100)
    time_np = timeit.Timer(lambda: gauss_np(a)).timeit(number=100)
    time_det = timeit.Timer(lambda: det(b)).timeit(number=100)

    return n, time_gauss, time_np, time_det


def write_file(results):
    with open("results/timings.txt", "w") as f:
        writer = csv.writer(f, delimiter='\t\t')
        for result in results:
            writer.writerow(result)
    return None


if __name__ == "__main__":
    results = [time_mat(i) for i in range(2, 43, 20)]
    write_file(results)
