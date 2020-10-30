import numpy as np


def det(a):
    """
    Calculates the determinant of a square matrix 
    of arbitrary size.
    
    Parameters
    ----------
    a : np.array or list of lists
        'n x n' array

    Examples
    --------
    >> a = [[2, 0, -1], [0, 5, 6], [0, -1, 1]]
    >> det(a)
    22.0
    >> A = [[1, 0, -1], [-2, 3, 0], [1, -3, 2]]
    >> det(A)
    3.0
    """
    det3 = 0

    #a = a.tolist()

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
