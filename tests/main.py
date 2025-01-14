import numpy as np
import scipy as sp
from PIL import Image
import argparse

from random_matrix import generate_matrix
from random_matrix import multiply_by_random_matrix

from scipy.optimize import linprog

def l1_minimization(A, y):
    m, n = A.shape

    # Construct the linear programming problem for L1 minimization
    c = np.concatenate([np.zeros(n), np.ones(m)])  # Objective: Minimize sum of z_i
    
    y = np.array(y)

    # Construct the linear programming problem for L1 norm minimization
    c = np.concatenate([np.ones(n), np.ones(n)])  # Objective: Minimize sum of u + v

    # Constraints: A(u - v) = y
    A_eq = np.hstack([A, A])
    b_eq = y

    # Bounds for u and v: u, v >= 0
    bounds = [(0, None)] * (2 * n)

    # Solve the linear program
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        u = result.x[:n]  # Extract u
        v = result.x[n:]  # Extract v
        return u - v  # Reconstruct x = u - v
    else:
        raise ValueError("L1 norm minimization failed to converge.")

def main(args):
    img = Image.open(args.path)
    img.show()
    array : np.ndarray = np.asarray(Image.open(args.path))
    
    #print(array.flatten())

    vector = array.flatten()
    N : int = vector.shape[0]
    s : int = N/2
    m = (int)(2*s*np.log(N/s))

    # too big
    #compression_matrix = generate_matrix(1,((int)(s*np.log(N/s)),N))
    #compressed_vector = np.matmul(compression_matrix,vector)
    compressed_vector = multiply_by_random_matrix(vector,1,(m,N))

    print(f"Compressed {N} to {m}")
    result = l1_minimization(generate_matrix(1,(m,N)),compressed_vector)

    img : Image = Image.fromarray(result.reshape(array.shape),mode="RGB")
    img.save('decompressed.png')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str)
    args = parser.parse_args()
    main(args)