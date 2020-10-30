def det(a):

    d = 0

    if(len(a) == 2):
        value = a[0][0]*a[1][1] - a[0][1]*a[1][0]
        return value

    for col in range(len(a)):

        ab = [r[:col] + r[col+1:] for r in (a[:0] + a[1:])]
                
        if not ab:
            
            continue
                             
        d = d + (-1)**(0 + col)*det(ab)*a[0][col]

    return d
