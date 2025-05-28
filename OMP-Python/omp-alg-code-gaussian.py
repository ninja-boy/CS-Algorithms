def measurement(m,n):

    # Create a measurement matrix
    A= np.random.randn(m, n)
    return A

def sparse(k,n):

    # Create a sparse matrix
    x= np.zeros(n)

    # Randomly select k indices to be non-zero
    nonzero_indices= np.random.choice(n, k, replace=False)
    x[nonzero_indices]= np.random.randn(k)
    return x

def omp(y, A, tol=1e-6):
    
    # Orthogonal Matching Pursuit (OMP) algorithm
    m,n =A.shape  # Get dimensions of A
    r =y.copy() # Residual initialisation
    idx_set =[]  # Indices of selected columns
    x_hat =np.zeros(n)  # Estimated sparse signal
    A_selected =[]  # Selected columns of A

    for _ in range(m):
        correlations= A.T @ r # Compute correlations between residual and columns of A transpose
        idx= np.argmax(np.abs(correlations))
        if idx not in idx_set:
            idx_set.append(idx)
        # Select columns of A corresponding to the indices in idx_set
        A_selected= A[:, idx_set]
        # Solve least squares problem to find the best fit
        x_ls, _, _, _= np.linalg.lstsq(A_selected, y, rcond=None)
        # Update the residual
        r= y - A_selected @ x_ls

    # Fill in the estimated sparse signal
    x_hat[idx_set]= x_ls
    return x_hat


#Main program
import numpy as np
from sys import exit
import matplotlib.pyplot as plt

len= int(input("Enter length of signal: "))
sp_val= int(input("Enter no of non zero elements: "))

#If sparsity is greater than length, exit with error
if sp_val>len:
    print("Error: Number of non-zero elements cannot be greater than the length of the signal.")
    exit(1)
sparse_signal= sparse(sp_val, len)
samples= int(input("Enter no of samples: "))
measurement_matrix= measurement(samples, len)
compressed_matrix= measurement_matrix @ sparse_signal

# OMP reconstruction
reconstructed_signal = omp(compressed_matrix, measurement_matrix)

#Compare original and reconstructed signals
print("True signal:-\n", sparse_signal)
print("Reconstructed:-\n", reconstructed_signal)
print("Are they close?    ", np.allclose(sparse_signal, reconstructed_signal, atol=1e-2))

# Plotting
plt.figure(figsize=(16, 9))
plt.stem(sparse_signal, linefmt='b-', markerfmt='bo', basefmt=' ', label='Original Signal')
plt.stem(reconstructed_signal, linefmt='r--', markerfmt='ro', basefmt=' ', label='Reconstructed Signal')
plt.title('Original vs Reconstructed Sparse Signal')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.legend()

# Add x-axis (y=0) and y-axis (x=0)
plt.axhline(0, color='black', linewidth=0.8)  # x-axis
plt.axvline(0, color='black', linewidth=0.8)  # y-axis

plt.show()
