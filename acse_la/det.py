import numpy as np


def det(a):

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
