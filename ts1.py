import numpy as np
import sympy as sp
from sympy.abc import t

def taylor_series_expansion(matrix, terms, t):
    n = matrix.shape[0]
    identity = np.eye(n)
    result = identity.copy()
    matrix_power = matrix.copy()


    for i in range(1, terms + 1):
        term = (matrix_power**i) * (t**i) * (1j**i) / np.math.factorial(i)
        result += term

    return result

# Example usage
#A = np.array([[1, 2],
#               [3, 4]])
#result = taylor_series_expansion(A, 2, 0.5)
#print(result)
#print(result.dtype)