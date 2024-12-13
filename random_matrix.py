
import numpy as np

def generate_matrix(seed = 1,dimensions = (100,100)):
    np.random.seed(seed)
    matrix = np.random.normal(0,1,dimensions)
    return matrix

def multiply_by_random_matrix(vector, seed = 1, dimensions = (100,100)):
    result = np.zeros((dimensions[0]))
    np.random.seed(seed)
    for i in np.arange(len(result)):
        matrix_row = np.random.normal(0, 1,dimensions[1])
        result[i] = np.dot(matrix_row,vector)
    return result

if __name__ == "__main__":
    print(generate_matrix())


