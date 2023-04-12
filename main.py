from vectors_examples import *
from linear_algebra import *

import numpy as np


def main():
    # a = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
    # print(reduced_row_echelon_form(a))
    # a = np.array([[7, 5, 3], [3, 2, 5], [1, 2, 1]])
    # c = np.array([120, 70, 20])
    # print(solve_systems_of_equations(a, c))
    # print(determinant(a))
    # print(matrix_rank(a))

    # j = np.array([[0, 1, 1], [2, 4, 2], [1, 2, 1]])
    # k = np.array([[7.5, 5, 12.5], [3, 2, 5], [0, 0, 0]])
    # l = np.array([[7, 5, 3], [3, 2, 5], [1, 2, 1]])

    # print(matrix_rank(j))
    # print(matrix_rank(k))
    # print(matrix_rank(l))

    # j = np.array([[5, 2], [10, 3]])
    # k = np.array([[0, 0], [0, 0]])
    # l = np.array([[1, 1], [2, 2]])

    # print(matrix_rank(j))
    # print(matrix_rank(k))
    # print(matrix_rank(l))

    # from exam
    A = np.array(
        [[2, -1, 1, 1], [1, 2, -1, -1], [-1, 2, 2, 2], [1, -1, 2, 1]], dtype=np.dtype(float)
    )
    b = np.array([6, 3, 14, 8], dtype=np.dtype(float))
    A_system = np.hstack((A, b.reshape(4, 1)))

    A_ref = SwapRows(A_system, 0, 1)

    # multiply row 0 of the new matrix A_ref by -2 and add it to the row 1
    A_ref = AddRows(A_ref, 0, 1, -2)

    # add row 0 of the new matrix A_ref to the row 2, replacing row 2
    A_ref = AddRows(A_ref, 0, 2, 1)

    # multiply row 0 of the new matrix A_ref by -1 and add it to the row 3
    A_ref = AddRows(A_ref, 0, 3, -1)

    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref, 2, 3, 1)

    # swap row 1 and 3 of the new matrix A_ref
    A_ref = SwapRows(A_ref, 1, 3)

    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref, 2, 3, 1)

    # multiply row 1 of the new matrix A_ref by -4 and add it to the row 2
    A_ref = AddRows(A_ref, 1, 2, -4)

    # add row 1 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref, 1, 3, 1)

    # multiply row 3 of the new matrix A_ref by 2 and add it to the row 2
    A_ref = AddRows(A_ref, 3, 2, 2)

    # multiply row 2 of the new matrix A_ref by -8 and add it to the row 3
    A_ref = AddRows(A_ref, 2, 3, -8)

    # multiply row 3 of the new matrix A_ref by -1/17
    A_ref = MultiplyRow(A_ref, 3, -1 / 17)
    print(A_ref)


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
