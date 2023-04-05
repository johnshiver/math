import numpy as np


def solve_systems_of_equations(coefficient_matrix, constant_matrix):
    return np.linalg.solve(coefficient_matrix, constant_matrix)


def determinant(matrix):
    return np.linalg.det(matrix)


def gaussian_elimination(A, b):
    """
    If you'd like to perform row operations and obtain the row echelon form,
    you can implement the Gaussian elimination algorithm in Python.
    Although this is not the most efficient method, it will show the row echelon
    form as an intermediate step.

    A = np.array([[1, 1, 1],
              [0, 2, 5],
              [2, 5, -1]])

    b = np.array([6, -4, 27])

    augmented_reduced = gaussian_elimination(A, b)
    print(augmented_reduced)
    """
    augmented = np.hstack((A, b.reshape(-1, 1)))
    n = len(augmented)

    for i in range(n):
        max_element_index = abs(augmented[i:, i]).argmax() + i

        if augmented[max_element_index, i] == 0:
            raise ValueError("Matrix is singular.")

        if max_element_index != i:
            augmented[[i, max_element_index]] = augmented[[max_element_index, i]]

        augmented[i] = augmented[i] / augmented[i, i]

        for j in range(i + 1, n):
            augmented[j] -= augmented[j, i] * augmented[i]

    return augmented


def matrix_rank(matrix):
    return np.linalg.matrix_rank(matrix)


def row_echelon_form(matrix):
    matrix = matrix.astype(float)
    n_rows, n_cols = matrix.shape

    row = 0
    for col in range(n_cols):
        if row >= n_rows:
            break

        pivot = matrix[row, col]
        if pivot == 0:
            nonzero_row = None
            for r in range(row + 1, n_rows):
                if matrix[r, col] != 0:
                    nonzero_row = r
                    break

            if nonzero_row is None:
                continue
            else:
                matrix[[row, nonzero_row]] = matrix[[nonzero_row, row]]
                pivot = matrix[row, col]

        for r in range(row + 1, n_rows):
            factor = matrix[r, col] / pivot
            matrix[r] -= factor * matrix[row]

        row += 1

    return matrix
