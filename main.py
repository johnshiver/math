from vectors_examples import *
from linear_algebra import *
from vectors import *

import numpy as np


def main():
    # u = (1, 3)
    # v = (6, 2)
    # print(add(u, v))
    # print(subtract(u, v))

    # a = np.array([-1, 5, 2])
    # b = np.array([-3, 6, -4])
    # print(a.dot(b))

    a = np.array([1, 2, -3])
    b = np.array([0, 0, 0, 0])
    c = np.array([1, 0, -2, 0, -1])
    d = np.array([2, 2, 2, 2])
    e = np.array([2, 5])

    print(np.linalg.norm(a))
    print(np.linalg.norm(b))
    print(np.linalg.norm(c))
    print(np.linalg.norm(d))
    print(np.linalg.norm(e))

    x = np.array([-1, 5, 2])
    y = np.array([-3, 6, -4])
    print(x.dot(y))

    m1 = np.array([[2, -1], [3, -3]])
    m2 = np.array([[5, -2], [0, 1]])

    print(m1 @ m2)

    x = np.array([-9, -1])
    y = np.array([-3, -5])
    print(x.dot(y))


def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    # exchange row_num_1 and row_num_2 of the matrix M_new
    row_1 = M[row_num_1]
    row_2 = M[row_num_2]
    M_new[row_num_1] = row_2
    M_new[row_num_2] = row_1
    return M_new


def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()
    # exchange row_num of the matrix M_new with its multiple by row_num_multiple
    # Note: for simplicity, you can drop check if  row_num_multiple has non-zero value, which makes the operation valid
    M_new[row_num] = row_num_multiple * M_new[row_num]
    return M_new


def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()
    # multiply row_num_1 by row_num_1_multiple and add it to the row_num_2,
    # exchanging row_num_2 of the matrix M_new with the result
    M_new[row_num_2] = (row_num_1_multiple * M_new[row_num_1]) + M_new[row_num_2]
    return M_new


if __name__ == "__main__":
    main()


"""
7f + 5a + 3c = 120
3f + 2a + 5c = 70
1f + 2a + 1c = 20
"""
