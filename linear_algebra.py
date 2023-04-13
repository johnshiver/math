import numpy as np

import matplotlib.pyplot as plt


def plot_lines(m):
    x_1 = np.linspace(-10, 10, 100)
    x_2_line_1 = (m[0, 2] - m[0, 0] * x_1) / m[0, 1]
    x_2_line_2 = (m[1, 2] - m[1, 0] * x_1) / m[1, 1]

    _, ax = plt.subplots(figsize=(10, 10))
    ax.plot(
        x_1,
        x_2_line_1,
        "-",
        linewidth=2,
        color="#0075ff",
        label=f"$x_2={-m[0,0]/m[0,1]:.2f}x_1 + {m[0,2]/m[0,1]:.2f}$",
    )
    ax.plot(
        x_1,
        x_2_line_2,
        "-",
        linewidth=2,
        color="#ff7300",
        label=f"$x_2={-m[1,0]/m[1,1]:.2f}x_1 + {m[1,2]/m[1,1]:.2f}$",
    )

    A = m[:, 0:-1]
    b = m[:, -1::].flatten()
    d = np.linalg.det(A)

    if d != 0:
        solution = np.linalg.solve(A, b)
        ax.plot(
            solution[0],
            solution[1],
            "-o",
            mfc="none",
            markersize=10,
            markeredgecolor="#ff0000",
            markeredgewidth=2,
        )
        ax.text(
            solution[0] - 0.25,
            solution[1] + 0.75,
            f"$(${solution[0]:.0f}$,{solution[1]:.0f})$",
            fontsize=14,
        )
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))

    plt.xlabel("$x_1$", size=14)
    plt.ylabel("$x_2$", size=14)
    plt.legend(loc="upper right", fontsize=14)
    plt.axis([-10, 10, -10, 10])

    plt.grid()
    plt.gca().set_aspect("equal")

    plt.show()


def solve_systems_of_equations(coefficient_matrix, constant_matrix):
    return np.linalg.solve(coefficient_matrix, constant_matrix)


def determinant(matrix):
    return np.linalg.det(matrix)


# Define a custom error class
class ErrMatrixCannotBeInverted(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def calculate_inverse(matrix):
    """
    Inverse of a matrix is like the inverse of a number. The inverse of a number multiplied
    by itself is 1. e.g. 2 * 1/2 = 1

    The inverse of a matrix multiplied by itself is the identity matrix.

    However, if the determinant of a matrix is 0, it is not invertible and an error is raised.
    """
    matrix = matrix.copy()
    det = determinant(matrix)

    if det == 0:
        raise ErrMatrixCannotBeInverted("matrix is not invertible because its determinant is 0")

    return np.linalg.inv(matrix)


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
    """
    From ChatGPT 4:

    The rank of a matrix is defined as the maximum number of linearly independent rows or columns in the matrix.

    Here are some key points regarding the rank:

    - It represents the dimension of the column space (also called the range) and the row space of the matrix.

    - The rank of a matrix can be used to determine the solvability of a system of linear equations.
      If the rank of the coefficient matrix (A) is equal to the rank of the augmented matrix ([A|b]),
      then the system has a solution. If the rank is less than the number of unknowns,
      there are infinitely many solutions. If the rank of A is less than the rank of the augmented matrix,
      the system has no solution.

    - The rank can help identify linear dependencies among the rows or columns of a matrix.
      If a matrix has full rank (i.e., the rank is equal to the number of rows or columns, whichever is smaller),
      all rows and columns are linearly independent, and the matrix is invertible if it is square.

    - The rank of a product of two matrices (A * B) cannot exceed the rank of either A or B.

    Applications of matrix rank include:

    - Solving systems of linear equations: The rank helps determine the existence and uniqueness of solutions.

    - Linear transformations: In the context of linear transformations, the rank is the dimension of the output space
      (i.e., the image of the transformation). It helps analyze the properties of the transformation, such as injectivity and surjectivity.

    - Data analysis and dimensionality reduction: In data analysis, rank can be used to identify linear relationships
      between variables and reduce the dimensionality of data. Principal component analysis (PCA), for instance,
      uses the rank and the eigenvectors of the covariance matrix to project data onto a lower-dimensional space.

    - Matrix approximations: In some applications, it's useful to approximate a matrix by a lower-rank matrix. For example,
      singular value decomposition (SVD) can be used to approximate a matrix by retaining only the most significant singular
      values and vectors, reducing storage and computation costs while preserving the most important information.

    """
    return np.linalg.matrix_rank(matrix)


def row_echelon_form(matrix):
    """
    a = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
    print(reduced_row_echelon_form(a))

    [[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]]


    From chatGPT:

    Transforming a matrix into its row echelon form (REF) can be beneficial for several reasons, particularly when dealing with systems of linear equations:

    - Simplification: The row echelon form simplifies a matrix by having zeros below the diagonal,
                      making it easier to work with when solving systems of linear equations or
                      performing other operations. The triangular structure of the REF allows for
                      the efficient use of back-substitution to solve the linear system.

    - Rank determination: The row echelon form allows you to easily determine the rank of a matrix,
                          which is the number of non-zero rows in the REF. The rank is an important property,
                          as it tells you about the dimensions of the column space and the row space, as well
                          as the solvability of a linear system.

    - Linear dependence/independence: By transforming a matrix into its row echelon form, you can easily identify
                                      linearly dependent or independent rows (or columns) of a matrix. If a row in the
                                      REF is all zeros, it is linearly dependent on the other rows. Otherwise, it is linearly
                                      independent. This information is crucial in various applications, such as determining the
                                      basis for the row or column space of a matrix.

    - Consistency check: The row echelon form helps you determine if a system of linear equations is consistent (has at least one solution)
                         or inconsistent (has no solutions). If there is a row in the augmented matrix (the matrix with the constants from
                         the linear equations appended as a column) with all zeros except for the last element, the system is inconsistent.

    - Inverse calculation: The REF is a stepping stone to finding the reduced row echelon form (RREF), which is essential for
                           calculating the inverse of a square matrix. By row reducing a matrix augmented with the identity matrix,
                           you can obtain the inverse (if it exists) in the identity matrix's original location.

    """
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


def reduced_row_echelon_form(matrix):
    """
    From chatGPT:

    Both row echelon form (REF) and reduced row echelon form (RREF) are simplified
    versions of a matrix, achieved through a series of elementary row operations.
    However, there are some differences in the structure of the two forms:

    Row Echelon Form (REF):

        All zero rows (if any) are at the bottom of the matrix.
        The first non-zero element in a non-zero row (called the pivot or leading entry) is
        always strictly to the right of the pivot in the row above.
        The pivot in each non-zero row is 1. All elements below the pivot in the same column are 0.

    Reduced Row Echelon Form (RREF):

        The RREF satisfies all the conditions of the REF.
        Additionally, in the RREF, all elements above the pivot in the same column are also 0.
        In summary, the RREF is a more restrictive and simplified version of the REF. The RREF has
        zeros both above and below each pivot, whereas the REF only requires zeros below each pivot.

    The RREF is unique for a given matrix, meaning that a matrix can have only one RREF.
    However, a matrix can have multiple REFs depending on the sequence of row operations applied to it.
    The RREF is particularly useful for solving systems of linear equations and finding the inverse of a matrix.
    """
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

        matrix[row] /= pivot

        for r in range(n_rows):
            if r == row:
                continue

            factor = matrix[r, col]
            matrix[r] -= factor * matrix[row]

        row += 1

    return matrix
